#!/usr/bin/env python3
"""
batch_bd_cheng_attn_vs_model.py

Compare your custom model checkpoints (alpha{lam}lambda.pt) against all qualities
of compressai.zoo.cheng2020_attn for up to the first `max_images` images in IMAGE_FOLDER.

Outputs:
 - agg CSVs of average RD points
 - BD metrics CSV (per lambda vs cheng)
 - RD plots per comparison (PSNR & MS-SSIM)
"""
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

# compressai model (Cheng attention)
from compressai.zoo import cheng2020_attn

# ms-ssim
import pytorch_msssim as msssim

# user model + loss (must be importable)
from model import CompressionModel, rate_distortion_loss

# ----------------------- DEFAULTS -----------------------
IMAGE_FOLDER = "/dcs/large/u2157170/code/results"
CKPT_FOLDER = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
OUTPUT_DIR = "/dcs/large/u2157170/code/modelv2/batch_bd_results_cheng_attn"

# keep your original LAMBDAS; script selects last couple by default
LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 10000]
# which of the lambdas to compare? default: last two
SELECT_LAMBDAS = LAMBDAS[-2:]

CHENG_QUALITIES = list(range(1, 9))  # 1..8
MAX_IMAGES = 1000

# model params (adjust if necessary)
N = 128
M = 192
SPATIAL_PARAMS = False
MIN_NU = 2.0
MAX_NU = 100.0

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# If your environment actually provides pretrained weights for cheng2020_attn, set to True.
# Script will try pretrained=True and fall back automatically if unavailable.
FORCE_PRETRAINED = True

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

def make_strictly_increasing(arr: np.ndarray, tiny=1e-9) -> np.ndarray:
    a = arr.astype(float).copy()
    for i in range(1, len(a)):
        if a[i] <= a[i-1]:
            a[i] = a[i-1] + tiny
    return a

# BD metrics (PCHIP) - same as your original
def bd_metrics(R1, P1, R2, P2):
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

# -------------------- main evaluation --------------------
def evaluate_vs_cheng_attn(image_folder: str = IMAGE_FOLDER,
                           ckpt_folder: str = CKPT_FOLDER,
                           output_dir: str = OUTPUT_DIR,
                           all_lambdas: List[int] = LAMBDAS,
                           select_lambdas: List[int] = SELECT_LAMBDAS,
                           cheng_qualities: List[int] = CHENG_QUALITIES,
                           max_images: int = MAX_IMAGES,
                           force_pretrained: bool = FORCE_PRETRAINED):
    os.makedirs(output_dir, exist_ok=True)

    # gather images
    files = [p for p in glob.glob(os.path.join(image_folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = sorted(files)[:max_images]
    if len(files) == 0:
        raise RuntimeError(f"No images found in {image_folder}")
    print(f"Found {len(files)} images; evaluating up to {max_images}")

    # prepare accumulators
    model_metrics = {lam: [] for lam in select_lambdas}  # (bpp, psnr, mss)
    cheng_metrics = {q: [] for q in cheng_qualities}

    # preload cheng models
    cheng_models = {}
    for q in cheng_qualities:
        loaded = False
        try:
            ch = cheng2020_attn(quality=q, pretrained=True, metric="mse").to(device).eval()
            cheng_models[q] = ch
            loaded = True
            print(f"Loaded cheng2020_attn quality={q} (pretrained=True)")
        except Exception as e:
            if force_pretrained:
                print(f"[WARN] pretrained cheng2020_attn q={q} failed: {e} -- falling back to pretrained=False")
            try:
                ch = cheng2020_attn(quality=q, pretrained=False, metric="mse").to(device).eval()
                cheng_models[q] = ch
                loaded = True
                print(f"Loaded cheng2020_attn quality={q} (pretrained=False)")
            except Exception as e2:
                print(f"[ERROR] failed to instantiate cheng2020_attn q={q}: {e2}")
        if not loaded:
            # leave quality out (it will be skipped)
            continue

    # preload user's models for selected lambdas
    user_models = {}
    for lam in select_lambdas:
        ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] checkpoint missing for lambda={lam}: {ckpt_path} -- skipping this lambda")
            continue
        try:
            model = CompressionModel(N=N, M=M, spatial_params=SPATIAL_PARAMS, min_nu=MIN_NU, max_nu=MAX_NU).to(device)
            state = torch.load(ckpt_path, map_location=device)
            if "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
            model.eval()
            user_models[lam] = model
            print(f"Loaded user model for lambda={lam}")
        except Exception as e:
            print(f"[WARN] failed to load checkpoint {ckpt_path}: {e}")

    if len(user_models) == 0 or len(cheng_models) == 0:
        raise RuntimeError("No user models or no cheng models loaded; aborting.")

    start_time = time.time()
    # iterate images
    for idx, img_path in enumerate(files, 1):
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] skipping image {img_path}: {e}")
            continue
        x = tensor_from_pil(pil)  # [1,3,H,W]
        H, W = x.shape[2], x.shape[3]
        num_pixels = int(H * W)

        # run cheng models
        for q, ch in cheng_models.items():
            try:
                with torch.no_grad():
                    out = ch(x)
                    x_hat = out.get("x_hat", None)
                    if x_hat is None:
                        try:
                            out = ch(x, quant_mode="round")
                        except Exception:
                            pass
                        x_hat = out.get("x_hat", None)
                    if x_hat is None:
                        raise RuntimeError("cheng output missing x_hat")
                    x_hat = x_hat[:, :, :H, :W].clamp(0,1)
                    bpp = compute_bpp_from_out(out, num_pixels, x_tensor=x)
                    psnr = compute_psnr(x, x_hat)
                    mss = compute_ms_ssim(x, x_hat)
                    cheng_metrics[q].append((bpp, psnr, mss))
            except Exception as e:
                print(f"[WARN] cheng q={q} failed on image {idx}/{len(files)}: {e}")

        # run user models (selected lambdas)
        for lam, model in user_models.items():
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

        if idx % 50 == 0 or idx == len(files):
            elapsed = time.time() - start_time
            print(f"Processed {idx}/{len(files)} images (elapsed {elapsed:.1f}s)")

    # Aggregate averages
    model_summary = []
    for lam, vals in model_metrics.items():
        if len(vals) == 0:
            print(f"[WARN] no successful runs for lambda={lam}; skipping in summary.")
            continue
        bpps = np.array([v[0] for v in vals], dtype=float)
        psnrs = np.array([v[1] for v in vals], dtype=float)
        msss = np.array([v[2] for v in vals], dtype=float)
        model_summary.append({"lambda": lam, "count": len(vals), "bpp": bpps.mean(), "psnr": psnrs.mean(), "msssim": msss.mean()})
    df_model = pd.DataFrame(model_summary).sort_values("bpp").reset_index(drop=True)
    df_model.to_csv(os.path.join(output_dir, "agg_model_selected_lambdas_rd.csv"), index=False)

    cheng_summary = []
    for q, vals in cheng_metrics.items():
        if len(vals) == 0:
            print(f"[WARN] no successful runs for cheng q={q}; skipping in summary.")
            continue
        bpps = np.array([v[0] for v in vals], dtype=float)
        psnrs = np.array([v[1] for v in vals], dtype=float)
        msss = np.array([v[2] for v in vals], dtype=float)
        cheng_summary.append({"quality": q, "count": len(vals), "bpp": bpps.mean(), "psnr": psnrs.mean(), "msssim": msss.mean()})
    df_cheng = pd.DataFrame(cheng_summary).sort_values("bpp").reset_index(drop=True)
    df_cheng.to_csv(os.path.join(output_dir, "agg_cheng2020_attn_rd.csv"), index=False)

    # compute BD metrics per selected lambda against cheng
    bd_rows = []
    for lam in df_model["lambda"].values:
        model_row = df_model[df_model["lambda"] == lam].iloc[0]
        model_bpp = np.array([model_row["bpp"]])
        model_psnr = np.array([model_row["psnr"]])
        model_mss = np.array([model_row["msssim"]])

        # Use entire cheng curve (averaged RD points) for comparison
        ch_bpp = df_cheng["bpp"].values
        ch_psnr = df_cheng["psnr"].values
        ch_mss = df_cheng["msssim"].values

        # BD requires at least 2 points on each curve; if not, skip.
        if len(ch_bpp) < 2 or len(model_bpp) < 2:
            # If model has only one averaged point (single lambda) we still try BD by treating the model as single-point curve:
            # BD computation needs >1 points; instead compute relative differences at closest cheng point and note NaN for BD.
            bd_rate_psnr = float('nan')
            bd_psnr = float('nan')
            bd_rate_mss = float('nan')
            bd_mss = float('nan')
            print(f"[INFO] Not enough points to compute BD between lambda={lam} and cheng (need >=2 each).")
        else:
            try:
                bd_rate_psnr, bd_psnr = bd_metrics(model_bpp, model_psnr, ch_bpp, ch_psnr)
            except Exception as e:
                bd_rate_psnr, bd_psnr = float('nan'), float('nan')
                print(f"[WARN] BD PSNR failed for lambda={lam}: {e}")
            try:
                bd_rate_mss, bd_mss = bd_metrics(model_bpp, model_mss, ch_bpp, ch_mss)
            except Exception as e:
                bd_rate_mss, bd_mss = float('nan'), float('nan')
                print(f"[WARN] BD MS-SSIM failed for lambda={lam}: {e}")

        bd_rows.append({
            "lambda": lam,
            "bd_rate_psnr_pct": bd_rate_psnr,
            "bd_psnr_db": bd_psnr,
            "bd_rate_mss_pct": bd_rate_mss,
            "bd_mss_diff": bd_mss
        })

    df_bd = pd.DataFrame(bd_rows)
    df_bd.to_csv(os.path.join(output_dir, "bd_summary_selected_vs_cheng.csv"), index=False)
    print("Saved BD summary:", os.path.join(output_dir, "bd_summary_selected_vs_cheng.csv"))

    # Plot averaged RD curves (one plot with lines for each selected lambda and cheng)
    try:
        plt.figure(figsize=(8,6))
        # plot cheng curve
        plt.plot(df_cheng["bpp"].values, df_cheng["psnr"].values, "s--", label="cheng2020_attn (avg)")
        # plot selected lambdas
        for _, row in df_model.iterrows():
            plt.plot(row["bpp"], row["psnr"], "o", label=f"lambda={int(row['lambda'])}")
        plt.xscale("log")
        plt.xlabel("Bits per pixel (bpp) [log scale]")
        plt.ylabel("PSNR (dB)")
        plt.title("PSNR vs bpp (averaged over images) - selected lambdas vs cheng2020_attn")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rd_psnr_selected_vs_cheng_avg_logx.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARN] plotting PSNR RD failed: {e}")

    try:
        plt.figure(figsize=(8,6))
        plt.plot(df_cheng["bpp"].values, df_cheng["msssim"].values, "s--", label="cheng2020_attn (avg)")
        for _, row in df_model.iterrows():
            plt.plot(row["bpp"], row["msssim"], "o", label=f"lambda={int(row['lambda'])}")
        plt.xscale("log")
        plt.xlabel("Bits per pixel (bpp) [log scale]")
        plt.ylabel("MS-SSIM")
        plt.title("MS-SSIM vs bpp (averaged over images) - selected lambdas vs cheng2020_attn")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rd_msssim_selected_vs_cheng_avg_logx.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARN] plotting MS-SSIM RD failed: {e}")

    # cleanup: delete loaded models to free memory
    for m in user_models.values():
        try:
            del m
        except Exception:
            pass
    for m in cheng_models.values():
        try:
            del m
        except Exception:
            pass
    torch.cuda.empty_cache()

    return {"df_model": df_model, "df_cheng": df_cheng, "df_bd": df_bd}

# ---------------- run when executed ----------------
if __name__ == "__main__":
    results = evaluate_vs_cheng_attn(
        image_folder=IMAGE_FOLDER,
        ckpt_folder=CKPT_FOLDER,
        output_dir=OUTPUT_DIR,
        all_lambdas=LAMBDAS,
        select_lambdas=SELECT_LAMBDAS,
        cheng_qualities=CHENG_QUALITIES,
        max_images=MAX_IMAGES,
        force_pretrained=FORCE_PRETRAINED
    )
    print("Done. BD summary:")
    print(results["df_bd"].to_string(index=False))
