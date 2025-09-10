#!/usr/bin/env python3
"""
compare_to_bmshj.py

For a single input image and a folder of checkpoints named alpha{lam}lambda.pt,
compare each custom-model reconstruction to the closest CompressAI bmshj2018_hyperprior (MSE)
in bpp. Save reconstructions and pixel-difference heatmaps and a summary CSV.

Adjust DEFAULT_* paths/params as needed.
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

# compressai zoo
from compressai.zoo import bmshj2018_hyperprior

# ms-ssim
import pytorch_msssim as msssim

# your model
from model import CompressionModel, rate_distortion_loss

# -------------------------- DEFAULTS --------------------------
DEFAULT_IMAGE = "/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_02_RGB.png"
DEFAULT_CKPT_FOLDER = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
DEFAULT_OUTPUT_DIR = "/dcs/large/u2157170/code/modelv2/results&analysiswriteup_bmshj"
LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 10000]

# model param defaults (match your training)
DEFAULT_N = 128
DEFAULT_M = 192
DEFAULT_SPATIAL_PARAMS = False
DEFAULT_MIN_NU = 2.0
DEFAULT_MAX_NU = 100.0

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

# metrics
def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y, reduction='mean').item())

def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val:float=1.0) -> float:
    mse = compute_mse(x, y)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val * max_val) / mse)

def compute_ms_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    # attempt 3-level MS-SSIM module first; fallback to single-scale ssim if assertion
    try:
        ms_module = msssim.MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=3,
            weights=[0.3, 0.5, 0.2],  # 3-level weights
        ).to(x.device)
        return float(ms_module(x, y).item())
    except AssertionError:
        # image too small for multi-scale; fallback to ssim
        return float(msssim.ssim(x, y, data_range=1.0, size_average=True).item())

# difference heatmap (grayscale mean-abs-diff)
def save_diff_heatmap(x: torch.Tensor, y: torch.Tensor, out_path: str):
    # x,y are [1,3,H,W] in [0,1]
    diff = (x - y).abs().mean(1, keepdim=True)  # [1,1,H,W]
    maxv = float(diff.max().clamp(min=1e-8).item())
    diff = diff / maxv
    diff_img = TF.to_pil_image(diff.squeeze(0).cpu())
    diff_img.save(out_path)

# robust bpp extraction
def compute_bpp_from_out(out: Dict, orig_pixels: int, x_tensor: Optional[torch.Tensor]=None) -> float:
    # prefer nll_y/nll_z if available (these are bits per element)
    if "nll_y" in out and "nll_z" in out:
        total = float(out["nll_y"].sum().item() + out["nll_z"].sum().item())
        return total / float(orig_pixels)
    # compressai returns 'likelihoods' often as dict of tensors (probabilities)
    if "likelihoods" in out and isinstance(out["likelihoods"], dict):
        total_bits = 0.0
        for val in out["likelihoods"].values():
            if torch.is_tensor(val):
                eps = 1e-12
                total_bits += -torch.log2(val.clamp_min(eps)).sum().item()
        return total_bits / float(orig_pixels)
    # fallback: use rate_distortion_loss if x_tensor provided
    if x_tensor is not None:
        try:
            _, R, _ = rate_distortion_loss(out, x_tensor, lambda_rd=1.0, dist="mse")
            return float(R.item())
        except Exception:
            pass
    raise RuntimeError("Unable to compute bpp from model output. out keys: " + ", ".join(list(out.keys())))

# ---------------- main procedure ----------------
def process_single_image(image_path: str = DEFAULT_IMAGE,
                         ckpt_folder: str = DEFAULT_CKPT_FOLDER,
                         output_dir: str = DEFAULT_OUTPUT_DIR,
                         lambdas: List[int] = LAMBDAS,
                         N:int = DEFAULT_N, M:int = DEFAULT_M,
                         spatial_params:bool = DEFAULT_SPATIAL_PARAMS,
                         min_nu:float = DEFAULT_MIN_NU, max_nu:float = DEFAULT_MAX_NU):
    os.makedirs(output_dir, exist_ok=True)

    pil_orig = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(pil_orig).unsqueeze(0).to(device)
    H, W = x.shape[2], x.shape[3]
    num_pixels = int(H*W)

    summary_rows = []

    # pre-create bmshj models for qualities 1..8 (lazy load)
    bm_models = {}
    for q in range(1,9):
        try:
            bm = bmshj2018_hyperprior(quality=q, pretrained=True, metric="mse").to(device).eval()
            bm_models[q] = bm
        except Exception as e:
            print(f"[WARN] cannot load bmshj quality={q}: {e}")

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

        # forward for custom model
        with torch.no_grad():
            x_padded, pad_h, pad_w = pad_to_multiple_tensor(x, multiple=16)
            # try quant_mode 'round' if supported
            try:
                out = model(x_padded, quant_mode="round")
            except TypeError:
                out = model(x_padded)
            if "x_hat" not in out:
                raise RuntimeError("Model output missing 'x_hat' key")
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0,1)
            out["x_hat"] = x_hat

            try:
                model_bpp = compute_bpp_from_out(out, num_pixels, x_tensor=x)
            except Exception:
                _, Rtmp, _ = rate_distortion_loss(out, x, lambda_rd=1.0, dist="mse")
                model_bpp = float(Rtmp.item())

        print(f"model bpp: {model_bpp:.6f}")

        # model metrics
        model_mse = compute_mse(x, x_hat)
        model_psnr = compute_psnr(x, x_hat)
        model_msssim = compute_ms_ssim(x, x_hat)

        # save model recon & diff
        pil_recon = pil_from_tensor(x_hat)
        recon_path = os.path.join(output_dir, f"reconstruction_{lam}lambda.png")
        pil_recon.save(recon_path)

        diff_model_path = os.path.join(output_dir, f"diff_model_{lam}lambda.png")
        save_diff_heatmap(x, x_hat, diff_model_path)

        # find best bmshj quality by comparing bpp (iterate preloaded bm_models)
        best_q = None
        best_diff = float("inf")
        best_bm_out = None
        best_bm_bpp = None
        best_bm_recon = None

        for q, bm in bm_models.items():
            with torch.no_grad():
                # bm expects inputs in [0,1], same as x
                try:
                    # compressai returns dict with likelihoods etc
                    bm_out = bm(x)  # use default behavior (pretrained eval)
                except TypeError:
                    bm_out = bm(x, quant_mode="round")
                # compute bm bpp
                try:
                    bm_bpp = compute_bpp_from_out(bm_out, num_pixels, x_tensor=x)
                except Exception:
                    # fallback to rate_distortion_loss
                    _, Rtmp2, _ = rate_distortion_loss(bm_out, x, lambda_rd=1.0, dist="mse")
                    bm_bpp = float(Rtmp2.item())

                diff = abs(bm_bpp - model_bpp)
                if diff < best_diff:
                    best_diff = diff
                    best_q = q
                    best_bm_out = bm_out
                    best_bm_bpp = bm_bpp

        if best_q is None:
            print("[WARN] no bmshj models available; skipping bm comparison for this lambda")
            continue

        # get bm reconstruction and metrics
        bm_x_hat = best_bm_out.get("x_hat", None)
        if bm_x_hat is None:
            raise RuntimeError("bmshj output missing x_hat")
        bm_x_hat = bm_x_hat[:, :, :H, :W].clamp(0,1)

        bm_mse = compute_mse(x, bm_x_hat)
        bm_psnr = compute_psnr(x, bm_x_hat)
        bm_msssim = compute_ms_ssim(x, bm_x_hat)

        # save bm recon and diff
        bm_recon_path = os.path.join(output_dir, f"bmshj_q{best_q}_equiv_{lam}lambda.png")
        pil_bm = pil_from_tensor(bm_x_hat)
        pil_bm.save(bm_recon_path)

        diff_bm_path = os.path.join(output_dir, f"diff_bmshj_q{best_q}_{lam}lambda.png")
        save_diff_heatmap(x, bm_x_hat, diff_bm_path)

        print(f"Selected bmshj quality={best_q} (bpp={best_bm_bpp:.6f}, diff_to_model_bpp={best_diff:.6f})")

        summary_rows.append({
            "lambda": lam,
            "ckpt": ckpt_path,
            "model_bpp": model_bpp,
            "model_mse": model_mse,
            "model_psnr": model_psnr,
            "model_ms-ssim": model_msssim,
            "model_recon": recon_path,
            "model_diff": diff_model_path,
            "bmshj_quality": best_q,
            "bmshj_bpp": best_bm_bpp,
            "bmshj_mse": bm_mse,
            "bmshj_psnr": bm_psnr,
            "bmshj_ms-ssim": bm_msssim,
            "bmshj_recon": bm_recon_path,
            "bmshj_diff": diff_bm_path
        })

        # cleanup
        del model, out, x_hat
        torch.cuda.empty_cache()

    # save CSV
    df = pd.DataFrame(summary_rows)
    csv_out = os.path.join(output_dir, "bpp_equiv_summary_bmshj.csv")
    df.to_csv(csv_out, index=False)
    print("\nSaved summary CSV to", csv_out)
    return df

# ---------------- run ----------------
if __name__ == "__main__":
    print("Running BPP-equivalent comparison vs bmshj2018 (MSE) for image:", DEFAULT_IMAGE)
    df = process_single_image(
        image_path=DEFAULT_IMAGE,
        ckpt_folder=DEFAULT_CKPT_FOLDER,
        output_dir=DEFAULT_OUTPUT_DIR,
        lambdas=LAMBDAS,
        N=DEFAULT_N, M=DEFAULT_M,
        spatial_params=DEFAULT_SPATIAL_PARAMS,
        min_nu=DEFAULT_MIN_NU, max_nu=DEFAULT_MAX_NU
    )
    print(df)
