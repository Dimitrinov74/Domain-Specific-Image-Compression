#!/usr/bin/env python3

import os
import io
from typing import List, Optional, Tuple, Dict
import glob
import math
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

# ms-ssim
import pytorch_msssim as msssim

# user model + loss (must be importable)
from model import CompressionModel, rate_distortion_loss

# ----------------------- DEFAULTS -----------------------
IMAGE_FOLDER = "/dcs/large/u2157170/code/results"
CKPT_FOLDER = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
OUTPUT_DIR = "/dcs/large/u2157170/code/modelv2/batch_bd_results_jpeg"
LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 10000]
JPEG_QUALITIES = list(range(10, 101, 10))  # 10,20,...,100 (change if you want)
MAX_IMAGES = 1000

# model params (adjust if necessary)
N = 128
M = 192
SPATIAL_PARAMS = False
MIN_NU = 2.0
MAX_NU = 100.0

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# metrics
def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y, reduction='mean').item())

def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val:float=1.0) -> float:
    mse = compute_mse(x, y)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((max_val * max_val) / mse)

def compute_ms_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    try:
        ms_module = msssim.MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=3,
            weights=[0.3, 0.5, 0.2]
        ).to(x.device)
        return float(ms_module(x, y).item())
    except AssertionError:
        return float(msssim.ssim(x, y, data_range=1.0, size_average=True).item())

def compute_bpp_from_out(out: Dict, orig_pixels: int, x_tensor: Optional[torch.Tensor]=None) -> float:
    # prefer nll_y/nll_z if available (usually bits per element)
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
    raise RuntimeError("Unable to compute bpp from model output. out keys: " + ", ".join(list(out.keys())))

def compute_bpp_from_jpeg_bytes(jpeg_bytes: bytes, orig_pixels: int) -> float:
    # bits per pixel
    return (len(jpeg_bytes) * 8.0) / float(orig_pixels)

def make_strictly_increasing(arr: np.ndarray, tiny=1e-9) -> np.ndarray:
    a = arr.astype(float).copy()
    for i in range(1, len(a)):
        if a[i] <= a[i-1]:
            a[i] = a[i-1] + tiny
    return a

# ---------- BD metrics using PCHIP as before ----------
def bd_metrics(R1, P1, R2, P2):
    """
    Compute BD-Rate (%) and BD-PSNR/BD-quality between curve1 and curve2.
    R1,R2: bitrates (bpp)
    P1,P2: quality (PSNR in dB or MS-SSIM in linear)
    Returns (bd_rate_percent, bd_quality)
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    order1 = np.argsort(P1)
    order2 = np.argsort(P2)
    P1s, R1s = P1[order1], R1[order1]
    P2s, R2s = P2[order2], R2[order2]

    P1s = make_strictly_increasing(P1s, tiny=1e-9)
    P2s = make_strictly_increasing(P2s, tiny=1e-9)

    p_min = max(P1s.min(), P2s.min())
    p_max = min(P1s.max(), P2s.max())
    if p_max <= p_min:
        raise ValueError("No overlap in quality range between curves -- cannot compute BD metrics.")

    logR1 = np.log(R1s)
    logR2 = np.log(R2s)

    logR1 = make_strictly_increasing(logR1, tiny=1e-12)
    logR2 = make_strictly_increasing(logR2, tiny=1e-12)

    interp1 = PchipInterpolator(P1s, logR1)
    interp2 = PchipInterpolator(P2s, logR2)

    f = lambda p: np.exp(interp1(p)) - np.exp(interp2(p))
    integral_diff = quad(f, p_min, p_max)[0] / (p_max - p_min)

    avg_ref = quad(lambda p: np.exp(interp2(p)), p_min, p_max)[0] / (p_max - p_min)
    bd_rate = integral_diff / avg_ref * 100.0

    interpR1 = PchipInterpolator(logR1, P1s)
    interpR2 = PchipInterpolator(logR2, P2s)
    rmin = max(logR1.min(), logR2.min())
    rmax = min(logR1.max(), logR2.max())
    if rmax <= rmin:
        bd_quality = float('nan')
    else:
        bd_quality = quad(lambda r: interpR1(r) - interpR2(r), rmin, rmax)[0] / (rmax - rmin)

    return bd_rate, bd_quality

# -------------------- main evaluation function --------------------
def evaluate_dataset_vs_jpeg(image_folder: str = IMAGE_FOLDER,
                             ckpt_folder: str = CKPT_FOLDER,
                             output_dir: str = OUTPUT_DIR,
                             lambdas: List[int] = LAMBDAS,
                             jpeg_qualities: List[int] = JPEG_QUALITIES,
                             max_images: int = MAX_IMAGES):
    os.makedirs(output_dir, exist_ok=True)

    # gather image files (natural sort)
    files = [p for p in glob.glob(os.path.join(image_folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = sorted(files)[:max_images]
    if len(files) == 0:
        raise RuntimeError(f"No images found in {image_folder}")

    print(f"Found {len(files)} images; evaluating up to {max_images}")

    # prepare accumulators
    model_metrics = {lam: [] for lam in lambdas}         # (bpp, psnr, mss)
    jpeg_metrics = {q: [] for q in jpeg_qualities}      # (bpp, psnr, mss)

    start_time = time.time()
    for idx, img_path in enumerate(files, 1):
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] skipping image {img_path}: {e}")
            continue
        x = tensor_from_pil(pil)  # [1,3,H,W]
        H, W = x.shape[2], x.shape[3]
        num_pixels = int(H * W)

        # JPEG baselines
        for q in jpeg_qualities:
            try:
                jpeg_bytes = jpeg_bytes_for_quality(pil, quality=q)
                # decode JPEG back to tensor
                decoded = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                x_hat = tensor_from_pil(decoded)  # move to device
                # crop to original size (usually same)
                x_hat = x_hat[:, :, :H, :W].clamp(0,1)
                bpp = compute_bpp_from_jpeg_bytes(jpeg_bytes, num_pixels)
                psnr = compute_psnr(x, x_hat)
                mss = compute_ms_ssim(x, x_hat)
                jpeg_metrics[q].append((bpp, psnr, mss))
            except Exception as e:
                print(f"[WARN] JPEG q={q} failed on image {idx}/{len(files)}: {e}")

        # run custom model for each lambda
        for lam in lambdas:
            ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")
            if not os.path.isfile(ckpt_path):
                continue
            try:
                model = CompressionModel(N=N, M=M, spatial_params=SPATIAL_PARAMS, min_nu=MIN_NU, max_nu=MAX_NU).to(device)
                state = torch.load(ckpt_path, map_location=device)
                if "model" in state:
                    model.load_state_dict(state["model"])
                else:
                    model.load_state_dict(state)
                model.eval()
            except Exception as e:
                print(f"[WARN] failed to load checkpoint {ckpt_path}: {e}")
                continue

            try:
                with torch.no_grad():
                    x_padded, pad_h, pad_w = pad_to_multiple_tensor(x, multiple=16)
                    try:
                        out = model(x_padded, quant_mode="round")
                    except TypeError:
                        out = model(x_padded)
                    if "x_hat" not in out:
                        raise RuntimeError("model output missing x_hat")
                    x_hat = out["x_hat"][:, :, :H, :W].clamp(0,1)
                    out["x_hat"] = x_hat
                    bpp = compute_bpp_from_out(out, num_pixels, x_tensor=x)
                    psnr = compute_psnr(x, x_hat)
                    mss = compute_ms_ssim(x, x_hat)
                    model_metrics[lam].append((bpp, psnr, mss))
            except Exception as e:
                print(f"[WARN] model lam={lam} failed on image {idx}/{len(files)}: {e}")
            finally:
                del model
                torch.cuda.empty_cache()

        if idx % 50 == 0 or idx == len(files):
            elapsed = time.time() - start_time
            print(f"Processed {idx}/{len(files)} images (elapsed {elapsed:.1f}s)")

    # Now aggregate per-lambda and per-quality by averaging across images that succeeded
    model_summary = []
    for lam in lambdas:
        vals = model_metrics.get(lam, [])
        if len(vals) == 0:
            continue
        bpps = np.array([v[0] for v in vals], dtype=float)
        psnrs = np.array([v[1] for v in vals], dtype=float)
        msss = np.array([v[2] for v in vals], dtype=float)
        model_summary.append({
            "lambda": lam,
            "count": len(vals),
            "bpp": bpps.mean(),
            "psnr": psnrs.mean(),
            "msssim": msss.mean()
        })

    jpeg_summary = []
    for q in jpeg_qualities:
        vals = jpeg_metrics.get(q, [])
        if len(vals) == 0:
            continue
        bpps = np.array([v[0] for v in vals], dtype=float)
        psnrs = np.array([v[1] for v in vals], dtype=float)
        msss = np.array([v[2] for v in vals], dtype=float)
        jpeg_summary.append({
            "quality": q,
            "count": len(vals),
            "bpp": bpps.mean(),
            "psnr": psnrs.mean(),
            "msssim": msss.mean()
        })

    df_model = pd.DataFrame(model_summary).sort_values("bpp").reset_index(drop=True)
    df_jpeg = pd.DataFrame(jpeg_summary).sort_values("bpp").reset_index(drop=True)

    # Save aggregated RD points
    df_model.to_csv(os.path.join(output_dir, "agg_model_rd.csv"), index=False)
    df_jpeg.to_csv(os.path.join(output_dir, "agg_jpeg_rd.csv"), index=False)
    print("Saved aggregated RD CSVs in", output_dir)

    # Convert to arrays for BD computation
    model_bpp = df_model["bpp"].values
    model_psnr = df_model["psnr"].values
    model_mss = df_model["msssim"].values

    jpeg_bpp = df_jpeg["bpp"].values
    jpeg_psnr = df_jpeg["psnr"].values
    jpeg_mss = df_jpeg["msssim"].values

    # --------- Compute BD metrics ---------
    bd_results = {}
    try:
        bd_rate_psnr, bd_psnr = bd_metrics(model_bpp, model_psnr, jpeg_bpp, jpeg_psnr)
    except Exception as e:
        bd_rate_psnr, bd_psnr = float('nan'), float('nan')
        print("Warning BD PSNR failed:", e)
    try:
        bd_rate_mss, bd_mss = bd_metrics(model_bpp, model_mss, jpeg_bpp, jpeg_mss)
    except Exception as e:
        bd_rate_mss, bd_mss = float('nan'), float('nan')
        print("Warning BD MS-SSIM failed:", e)

    bd_results["bd_rate_psnr_pct"] = bd_rate_psnr
    bd_results["bd_psnr_db"] = bd_psnr
    bd_results["bd_rate_mss_pct"] = bd_rate_mss
    bd_results["bd_mss_diff"] = bd_mss

    # Save BD results to CSV
    pd.DataFrame([bd_results]).to_csv(os.path.join(output_dir, "bd_metrics_summary_jpeg.csv"), index=False)
    print("Saved BD metrics to", os.path.join(output_dir, "bd_metrics_summary_jpeg.csv"))

    # --------- Plot RD curves (log x-axis as requested) ---------
    # PSNR vs bpp
    plt.figure(figsize=(8,6))
    plt.plot(model_bpp, model_psnr, "o-", label="My model (PSNR)")
    plt.plot(jpeg_bpp, jpeg_psnr, "s--", label="JPEG (PSNR)")
    plt.xscale("log")
    plt.xlabel("Bits per pixel (bpp) [log scale]")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR vs bpp (averaged over images)\nBD-Rate: {bd_rate_psnr:.2f}%, BD-PSNR: {bd_psnr:.3f} dB")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rd_psnr_avg_logx_jpeg.png"), dpi=300)
    plt.close()

    # MS-SSIM vs bpp
    plt.figure(figsize=(8,6))
    plt.plot(model_bpp, model_mss, "o-", label="My model (MS-SSIM)")
    plt.plot(jpeg_bpp, jpeg_mss, "s--", label="JPEG (MS-SSIM)")
    plt.xscale("log")
    plt.xlabel("Bits per pixel (bpp) [log scale]")
    plt.ylabel("MS-SSIM")
    plt.title(f"MS-SSIM vs bpp (averaged over images)\nBD-Rate: {bd_rate_mss:.2f}%, BD-MS-SSIM: {bd_mss:.4f}")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rd_msssim_avg_logx_jpeg.png"), dpi=300)
    plt.close()

    print("Saved RD plots in", output_dir)
    return {
        "df_model": df_model,
        "df_jpeg": df_jpeg,
        "bd_metrics": bd_results
    }

# ---------------- run when executed ----------------
if __name__ == "__main__":
    results = evaluate_dataset_vs_jpeg(
        image_folder=IMAGE_FOLDER,
        ckpt_folder=CKPT_FOLDER,
        output_dir=OUTPUT_DIR,
        lambdas=LAMBDAS,
        jpeg_qualities=JPEG_QUALITIES,
        max_images=MAX_IMAGES
    )
    print("Done. BD metrics:")
    print(results["bd_metrics"])
