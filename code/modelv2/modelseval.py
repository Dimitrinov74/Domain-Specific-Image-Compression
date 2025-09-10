#!/usr/bin/env python3
"""
batch_eval_model_rd.py

Evaluate your model checkpoints alpha{lam}lambda.pt on the first `max_images` images in IMAGE_FOLDER.
For each lambda:
 - compute per-image: bpp, MSE, PSNR, MS-SSIM
 - compute averages across images
 - save per-image CSV and aggregated summary CSV
 - save RD plots (PSNR vs bpp and MS-SSIM vs bpp) (linear & log x-axis)

Adjust defaults below if necessary.
"""
import os
import io
import glob
import math
import time
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scipy for BD not needed here, but keep if you want later
# from scipy.interpolate import PchipInterpolator
# from scipy.integrate import quad

# ms-ssim
import pytorch_msssim as msssim

# user model + loss (must be importable)
from model import CompressionModel, rate_distortion_loss

# ----------------------- CONFIG -----------------------
IMAGE_FOLDER = "/dcs/large/u2157170/code/results"
CKPT_FOLDER = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
OUTPUT_DIR = "/dcs/large/u2157170/code/modelv2/batch_eval_model"
LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 10000]
MAX_IMAGES = 1000

# model params (adjust if necessary)
N = 128
M = 192
SPATIAL_PARAMS = False
MIN_NU = 2.0
MAX_NU = 100.0

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

def tensor_from_pil(p: Image.Image) -> torch.Tensor:
    return TF.to_tensor(p).unsqueeze(0).to(device)

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
    # prefer returned nll_y/nll_z if present
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
            # rate_distortion_loss expected to return (_, rate, _)
            _, R, _ = rate_distortion_loss(out, x_tensor, lambda_rd=1.0, dist="mse")
            return float(R.item())
        except Exception:
            pass
    raise RuntimeError("Unable to compute bpp from model output. out keys: " + ", ".join(list(out.keys())))

# -------------------- evaluation --------------------
def evaluate_model_lambdas(image_folder=IMAGE_FOLDER,
                           ckpt_folder=CKPT_FOLDER,
                           lambdas: List[int]=LAMBDAS,
                           max_images: int=MAX_IMAGES,
                           output_dir: str=OUTPUT_DIR):
    # gather image files
    files = [p for p in glob.glob(os.path.join(image_folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = sorted(files)[:max_images]
    if len(files) == 0:
        raise RuntimeError(f"No images found in {image_folder}")
    print(f"Found {len(files)} images; evaluating up to {len(files)} images.")

    # global per-image-per-lambda list for saving
    detailed_rows = []

    # aggregate summary rows
    summary_rows = []

    start_all = time.time()
    # iterate lambdas and load model once per lambda
    for lam in lambdas:
        ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] checkpoint missing for lambda={lam}: {ckpt_path} -- skipping.")
            continue

        print(f"\n=== Evaluating lambda={lam} ===")
        # load model once
        try:
            model = CompressionModel(N=N, M=M, spatial_params=SPATIAL_PARAMS, min_nu=MIN_NU, max_nu=MAX_NU).to(device)
            state = torch.load(ckpt_path, map_location=device)
            if "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
            model.eval()
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {ckpt_path}: {e}")
            continue

        lam_bpps = []
        lam_psnrs = []
        lam_msses = []
        lam_mss = []

        start = time.time()
        for idx, img_path in enumerate(files, 1):
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] skipping image {img_path}: {e}")
                continue
            x = tensor_from_pil(pil)  # [1,3,H,W]
            H, W = x.shape[2], x.shape[3]
            num_pixels = int(H * W)

            try:
                with torch.no_grad():
                    x_padded, pad_h, pad_w = pad_to_multiple_tensor(x, multiple=16)
                    # call with quant_mode if model supports
                    try:
                        out = model(x_padded, quant_mode="round")
                    except TypeError:
                        out = model(x_padded)
                    if "x_hat" not in out:
                        raise RuntimeError("model output missing x_hat")
                    x_hat = out["x_hat"][:, :, :H, :W].clamp(0,1)

                    # compute bpp
                    bpp = compute_bpp_from_out(out, num_pixels, x_tensor=x)

                    # metrics
                    mse = compute_mse(x, x_hat)
                    psnr = compute_psnr(x, x_hat)
                    mss = compute_ms_ssim(x, x_hat)

                    lam_bpps.append(bpp)
                    lam_psnrs.append(psnr)
                    lam_msses.append(mse)
                    lam_mss.append(mss)

                    # record per-image row
                    detailed_rows.append({
                        "lambda": lam,
                        "image": os.path.basename(img_path),
                        "bpp": bpp,
                        "mse": mse,
                        "psnr": psnr,
                        "msssim": mss
                    })
            except Exception as e:
                print(f"[WARN] lambda={lam} failed on image {idx}/{len(files)} ({img_path}): {e}")
                continue

            if idx % 100 == 0 or idx == len(files):
                elapsed = time.time() - start
                print(f"  processed {idx}/{len(files)} (elapsed {elapsed:.1f}s)")

        # averaging (guard empty)
        count = len(lam_bpps)
        if count == 0:
            print(f"[WARN] No successful outputs for lambda={lam}; skipping summary entry.")
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue

        mean_bpp = float(np.mean(lam_bpps))
        mean_psnr = float(np.mean(lam_psnrs))
        mean_mse = float(np.mean(lam_msses))
        mean_msssim = float(np.mean(lam_mss))

        summary_rows.append({
            "lambda": lam,
            "count": count,
            "bpp": mean_bpp,
            "mse": mean_mse,
            "psnr": mean_psnr,
            "msssim": mean_msssim
        })

        print(f"Lambda={lam} summary: count={count}, mean bpp={mean_bpp:.4f}, psnr={mean_psnr:.3f} dB, msssim={mean_msssim:.4f}")

        # free model
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()

    total_elapsed = time.time() - start_all
    print(f"\nFinished evaluation for all lambdas in {total_elapsed:.1f}s")

    # Save detailed per-image CSV
    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv = os.path.join(output_dir, "per_image_per_lambda_results.csv")
    df_detailed.to_csv(detailed_csv, index=False)
    print("Saved per-image results ->", detailed_csv)

    # Save aggregated summary CSV
    df_summary = pd.DataFrame(summary_rows).sort_values("bpp").reset_index(drop=True)
    summary_csv = os.path.join(output_dir, "agg_model_rd_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print("Saved aggregated summary ->", summary_csv)

    # Plot RD curves (averaged points: one point per lambda)
    if not df_summary.empty:
        bpp_vals = df_summary["bpp"].values
        psnr_vals = df_summary["psnr"].values
        mss_vals = df_summary["msssim"].values

        # PSNR vs bpp (linear x-axis)
        plt.figure(figsize=(8,6))
        plt.plot(bpp_vals, psnr_vals, "o-", label="Model (PSNR)")
        plt.xlabel("Bits per pixel (bpp)")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs bpp (averaged over images)")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(output_dir, "rd_psnr_avg_linearx.png")
        plt.savefig(p, dpi=300)
        plt.close()
        print("Saved:", p)

        # PSNR vs bpp (log x-axis)
        plt.figure(figsize=(8,6))
        plt.plot(bpp_vals, psnr_vals, "o-", label="Model (PSNR)")
        plt.xscale("log")
        plt.xlabel("Bits per pixel (bpp) [log scale]")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs bpp (averaged over images) [log x]")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(output_dir, "rd_psnr_avg_logx.png")
        plt.savefig(p, dpi=300)
        plt.close()
        print("Saved:", p)

        # MS-SSIM vs bpp (linear)
        plt.figure(figsize=(8,6))
        plt.plot(bpp_vals, mss_vals, "o-", label="Model (MS-SSIM)")
        plt.xlabel("Bits per pixel (bpp)")
        plt.ylabel("MS-SSIM")
        plt.title("MS-SSIM vs bpp (averaged over images)")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(output_dir, "rd_msssim_avg_linearx.png")
        plt.savefig(p, dpi=300)
        plt.close()
        print("Saved:", p)

        # MS-SSIM vs bpp (log x-axis)
        plt.figure(figsize=(8,6))
        plt.plot(bpp_vals, mss_vals, "o-", label="Model (MS-SSIM)")
        plt.xscale("log")
        plt.xlabel("Bits per pixel (bpp) [log scale]")
        plt.ylabel("MS-SSIM")
        plt.title("MS-SSIM vs bpp (averaged over images) [log x]")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        p = os.path.join(output_dir, "rd_msssim_avg_logx.png")
        plt.savefig(p, dpi=300)
        plt.close()
        print("Saved:", p)
    else:
        print("[INFO] No summary points to plot (df_summary empty).")

    return {"detailed_csv": detailed_csv, "summary_csv": summary_csv, "summary_df": df_summary}

# ---------------- run when executed ----------------
if __name__ == "__main__":
    results = evaluate_model_lambdas()
    print("\nDone.")
    print("Summary CSV:", results["summary_csv"])
    print("Per-image CSV:", results["detailed_csv"])
    if not results["summary_df"].empty:
        print(results["summary_df"].to_string(index=False))
