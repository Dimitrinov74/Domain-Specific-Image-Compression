#!/usr/bin/env python3
"""
Self-contained script (no CLI args) that:
 - loads a single image,
 - for each lambda checkpoint alpha{lam}lambda.pt in a folder:
     - loads model, runs reconstruction,
     - computes model bpp, MSE, PSNR, MS-SSIM,
     - finds a JPEG quality with approx same bpp, saves JPEG,
     - computes MSE/PSNR/MS-SSIM for JPEG,
     - saves a 3-column composite: Original | Model | JPEG,
     - saves pixel-difference heatmaps (Original-Model, Original-JPEG),
 - saves bpp_equiv_summary.csv in the output folder.
"""
import os
import io
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
import pytorch_msssim as msssim

# import your model + loss
from model import CompressionModel, rate_distortion_loss

# -------------------------- DEFAULTS --------------------------
DEFAULT_IMAGE = "/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_02_RGB.png"
DEFAULT_CKPT_FOLDER = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
DEFAULT_OUTPUT_DIR = "/dcs/large/u2157170/code/modelv2/results&analysiswriteup"
LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 10000]

# model param defaults
DEFAULT_N = 128
DEFAULT_M = 192
DEFAULT_SPATIAL_PARAMS = False
DEFAULT_MIN_NU = 2.0
DEFAULT_MAX_NU = 100.0

# JPEG search params
JPEG_Q_MIN = 1
JPEG_Q_MAX = 95
JPEG_TOL_REL = 0.01
JPEG_MAX_ITERS = 12

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- helpers --------------------
def pad_to_multiple_tensor(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor,int,int]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, pad_h, pad_w

def pil_from_tensor(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().squeeze(0).clamp(0,1)
    return TF.to_pil_image(t)

def tensor_from_pil(p: Image.Image) -> torch.Tensor:
    return TF.to_tensor(p).unsqueeze(0).to(device)

def jpeg_bytes_for_quality(pil_img: Image.Image, quality: int) -> bytes:
    bio = io.BytesIO()
    pil_img.save(bio, format="JPEG", quality=int(quality), optimize=True)
    return bio.getvalue()

def jpeg_bpp_for_quality(pil_img: Image.Image, quality: int, num_pixels: int) -> Tuple[float,int]:
    b = jpeg_bytes_for_quality(pil_img, quality)
    return (len(b) * 8) / float(num_pixels), len(b)

def find_jpeg_quality_for_target_bpp(pil_img: Image.Image, target_bpp: float, num_pixels: int,
                                     q_min:int=JPEG_Q_MIN, q_max:int=JPEG_Q_MAX,
                                     tol_rel:float=JPEG_TOL_REL, max_iters:int=JPEG_MAX_ITERS) -> Tuple[int,float,int]:
    q_lo, q_hi = q_min, q_max
    bpp_lo, bytes_lo = jpeg_bpp_for_quality(pil_img, q_lo, num_pixels)
    bpp_hi, bytes_hi = jpeg_bpp_for_quality(pil_img, q_hi, num_pixels)

    if target_bpp <= bpp_lo:
        return q_lo, bpp_lo, bytes_lo
    if target_bpp >= bpp_hi:
        return q_hi, bpp_hi, bytes_hi

    lo, hi = q_lo, q_hi
    best_q, best_bpp, best_bytes = lo, bpp_lo, bytes_lo
    best_diff = abs(bpp_lo - target_bpp)

    for _ in range(max_iters):
        mid = (lo + hi) // 2
        bpp_mid, bytes_mid = jpeg_bpp_for_quality(pil_img, mid, num_pixels)
        diff = abs(bpp_mid - target_bpp)
        if diff < best_diff:
            best_diff, best_q, best_bpp, best_bytes = diff, mid, bpp_mid, bytes_mid
        if abs(bpp_mid - target_bpp) / max(target_bpp, 1e-9) <= tol_rel:
            return mid, bpp_mid, bytes_mid
        if bpp_mid < target_bpp:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best_q, best_bpp, best_bytes

# metrics
def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y, reduction='mean').item())

def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val:float=1.0) -> float:
    mse = compute_mse(x, y)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val * max_val) / mse)

def compute_ms_ssim(x, y) -> float:
    try:
        ms_module = msssim.MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=3,
            weights=[0.3, 0.5, 0.2]
        ).to(x.device)
        return ms_module(x, y).item()
    except AssertionError:
        return msssim.ssim(x, y, data_range=1.0, size_average=True).item()

# difference heatmap
def save_diff_heatmap(x: torch.Tensor, y: torch.Tensor, out_path: str):
    diff = (x - y).abs().mean(1, keepdim=True)  # [1,1,H,W]
    diff = diff / diff.max().clamp(min=1e-8)
    diff_img = TF.to_pil_image(diff.squeeze(0).cpu())
    diff_img.save(out_path)

# robust bpp extraction
def compute_bpp_from_out(out: Dict, orig_pixels: int, x_tensor: Optional[torch.Tensor]=None) -> float:
    if "nll_y" in out and "nll_z" in out:
        total = float(out["nll_y"].sum().item() + out["nll_z"].sum().item())
        return total / float(orig_pixels)
    if "likelihoods" in out and isinstance(out["likelihoods"], dict):
        total_bits = 0.0
        for val in out["likelihoods"].values():
            if torch.is_tensor(val):
                eps = 1e-12
                total_bits += -torch.log2(val.clamp_min(eps)).sum().item()
        return total_bits / float(orig_pixels)
    if x_tensor is not None:
        try:
            _, R, _ = rate_distortion_loss(out, x_tensor, lambda_rd=1.0, dist="mse")
            return float(R.item())
        except Exception:
            pass
    raise RuntimeError("Unable to compute bpp from model output.")

# ---------------- main procedure ----------------
def process_single_image(image_path: str = DEFAULT_IMAGE,
                         ckpt_folder: str = DEFAULT_CKPT_FOLDER,
                         output_dir: str = DEFAULT_OUTPUT_DIR,
                         lambdas: List[int] = LAMBDAS,
                         N:int = DEFAULT_N, M:int = DEFAULT_M,
                         spatial_params:bool = DEFAULT_SPATIAL_PARAMS,
                         min_nu:float = DEFAULT_MIN_NU, max_nu:float = DEFAULT_MAX_NU,
                         jpeg_tol_rel:float = JPEG_TOL_REL):
    os.makedirs(output_dir, exist_ok=True)

    pil_orig = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(pil_orig).unsqueeze(0).to(device)
    H, W = x.shape[2], x.shape[3]
    num_pixels = int(H*W)

    summary_rows = []

    for lam in lambdas:
        ckpt_name = f"alpha{lam}lambda.pt"
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] checkpoint missing: {ckpt_path}  (skipping)")
            continue

        print(f"\n--- lambda={lam} -> loading {ckpt_path} ---")
        model = CompressionModel(N=N, M=M, spatial_params=spatial_params, min_nu=min_nu, max_nu=max_nu).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

        with torch.no_grad():
            x_padded, pad_h, pad_w = pad_to_multiple_tensor(x, multiple=16)
            try:
                out = model(x_padded, quant_mode="round")
            except TypeError:
                out = model(x_padded)
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0,1)
            out["x_hat"] = x_hat

            try:
                model_bpp = compute_bpp_from_out(out, num_pixels, x_tensor=x)
            except Exception:
                _, Rtmp, _ = rate_distortion_loss(out, x, lambda_rd=1.0, dist="mse")
                model_bpp = float(Rtmp.item())

        print(f"model bpp: {model_bpp:.6f}")

        # metrics for model
        model_mse = compute_mse(x, x_hat)
        model_psnr = compute_psnr(x, x_hat)
        model_msssim = compute_ms_ssim(x, x_hat)

        pil_recon = pil_from_tensor(x_hat)
        recon_path = os.path.join(output_dir, f"reconstruction_{lam}lambda.png")
        pil_recon.save(recon_path)

        diff_model_path = os.path.join(output_dir, f"diff_model_{lam}lambda.png")
        save_diff_heatmap(x, x_hat, diff_model_path)

        best_q, jpeg_bpp, jpeg_bytes = find_jpeg_quality_for_target_bpp(pil_orig, model_bpp, num_pixels)
        print(f"JPEG q={best_q}, jpeg_bpp={jpeg_bpp:.6f}, bytes={jpeg_bytes}")

        jpeg_bytes_val = jpeg_bytes_for_quality(pil_orig, best_q)
        jpeg_path = os.path.join(output_dir, f"jpeg_equiv_{lam}lambda_q{best_q}.jpg")
        with open(jpeg_path, "wb") as f:
            f.write(jpeg_bytes_val)

        pil_jpeg = Image.open(io.BytesIO(jpeg_bytes_val)).convert("RGB")
        x_jpeg = tensor_from_pil(pil_jpeg)

        jpeg_mse = compute_mse(x, x_jpeg)
        jpeg_psnr = compute_psnr(x, x_jpeg)
        jpeg_msssim = compute_ms_ssim(x, x_jpeg)

        diff_jpeg_path = os.path.join(output_dir, f"diff_jpeg_{lam}lambda.png")
        save_diff_heatmap(x, x_jpeg, diff_jpeg_path)

        summary_rows.append({
            "lambda": lam,
            "ckpt": ckpt_path,
            "model_bpp": model_bpp,
            "model_mse": model_mse,
            "model_psnr": model_psnr,
            "model_ms-ssim": model_msssim,
            "model_recon": recon_path,
            "model_diff": diff_model_path,
            "jpeg_quality": best_q,
            "jpeg_bpp": jpeg_bpp,
            "jpeg_bytes": jpeg_bytes,
            "jpeg_mse": jpeg_mse,
            "jpeg_psnr": jpeg_psnr,
            "jpeg_ms-ssim": jpeg_msssim,
            "jpeg_path": jpeg_path,
            "jpeg_diff": diff_jpeg_path
        })

        del model, out, x_hat
        torch.cuda.empty_cache()

    df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(output_dir, "bpp_equiv_summary.csv")
    df.to_csv(csv_out, index=False)
    print("\nSaved summary CSV to", csv_out)
    return df

# ---------------- run ----------------
if __name__ == "__main__":
    print("Running BPP-equivalent comparison for image:", DEFAULT_IMAGE)
    df = process_single_image(
        image_path=DEFAULT_IMAGE,
        ckpt_folder=DEFAULT_CKPT_FOLDER,
        output_dir=DEFAULT_OUTPUT_DIR,
        lambdas=LAMBDAS,
        N=DEFAULT_N, M=DEFAULT_M,
        spatial_params=DEFAULT_SPATIAL_PARAMS,
        min_nu=DEFAULT_MIN_NU, max_nu=DEFAULT_MAX_NU,
        jpeg_tol_rel=JPEG_TOL_REL
    )
    print(df)

