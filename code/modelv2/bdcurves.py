import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from compressai.zoo import bmshj2018_hyperprior
import scipy.interpolate as si
from scipy.integrate import quad

device = "cuda" if torch.cuda.is_available() else "cpu"

def pad_to_multiple(x, multiple=16):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, pad_h, pad_w

def unpad(x, pad_h, pad_w):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

# --- Helpers ---
def load_image(path):
    img = Image.open(path).convert("RGB")
    x = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)
    return x

def psnr(x, y):
    mse = F.mse_loss(y, x, reduction="mean")
    return 10 * torch.log10(1.0 / mse)

def bpp_from_likelihoods(likelihoods, num_pixels):
    total_bits = 0.0
    for l in likelihoods.values():  # <-- iterate over the tensors, not keys
        total_bits += (-torch.log2(l).sum()).item()
    return total_bits / num_pixels

def test_model(model, x):
    x_padded, pad_h, pad_w = pad_to_multiple(x, multiple=16)
    with torch.no_grad():
        out = model(x_padded)
    num_pixels = x.size(2) * x.size(3)  # count original pixels for bpp
    bpp = bpp_from_likelihoods(out["likelihoods"], num_pixels)
    x_hat = out["x_hat"].clamp(0,1)
    x_hat = unpad(x_hat, pad_h, pad_w)   # crop back to original size
    return bpp, psnr(x, x_hat).item()

# def test_model_msssim(model, x):
#     """
#     Evaluate a single image and return (bpp, MS-SSIM)
#     """
#     x_padded, pad_h, pad_w = pad_to_multiple(x, multiple=16)
#     with torch.no_grad():
#         out = model(x_padded)

#     num_pixels = x.size(2) * x.size(3)
#     bpp = bpp_from_likelihoods(out["likelihoods"], num_pixels)
#     x_hat = out["x_hat"].clamp(0,1)
#     x_hat = unpad(x_hat, pad_h, pad_w)

#     # Compute MS-SSIM between original and reconstruction
#     import pytorch_msssim
#     ms_ssim_val = pytorch_msssim.ms_ssim(
#         x_hat, x, data_range=1.0, size_average=True
#     ).item()
#     return bpp, ms_ssim_val
import torch
import torch.nn.functional as F
import pytorch_msssim

def test_model_msssim(model, x, min_size=160):
    """
    Evaluate a single image with the model and return (bpp, MS-SSIM)
    Automatically upsamples small images to satisfy MS-SSIM minimum size.
    """
    # --- Pad to multiple of 16 ---
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    # --- Forward pass ---
    with torch.no_grad():
        out = model(x_padded, quant_mode="round")
    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)  # crop back

    # --- Upsample if too small ---
    h_hat, w_hat = x_hat.shape[2], x_hat.shape[3]
    if min(h_hat, w_hat) < min_size:
        scale = min_size / min(h_hat, w_hat)
        x_hat = F.interpolate(x_hat, scale_factor=scale, mode='bilinear', align_corners=False)
        x_resized = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
    else:
        x_resized = x

    # --- Compute MS-SSIM ---
    ms_val = pytorch_msssim.ms_ssim(x_hat, x_resized, data_range=1.0, size_average=True)

    # --- Compute BPP ---
    num_pixels = x.size(2) * x.size(3)
    total_bits = 0.0
    for l in out["likelihoods"].values():
        total_bits += (-torch.log2(l).sum()).item()
    bpp = total_bits / num_pixels

    return bpp, ms_val.item()


def bd_metrics(R1, P1, R2, P2):
    """Compute BD-Rate (%) and BD-PSNR (dB)."""
    order1, order2 = np.argsort(P1), np.argsort(P2)
    P1, R1 = np.array(P1)[order1], np.array(R1)[order1]
    P2, R2 = np.array(P2)[order2], np.array(R2)[order2]
    logR1, logR2 = np.log(R1), np.log(R2)

    interp1 = si.PchipInterpolator(P1, logR1)
    interp2 = si.PchipInterpolator(P2, logR2)

    pmin, pmax = max(min(P1), min(P2)), min(max(P1), max(P2))
    f = lambda p: np.exp(interp1(p)) - np.exp(interp2(p))
    bd_rate = quad(f, pmin, pmax)[0] / (pmax-pmin)
    bd_rate = bd_rate / (quad(lambda p: np.exp(interp2(p)), pmin, pmax)[0] / (pmax-pmin)) * 100

    interpR1 = si.PchipInterpolator(logR1, P1)
    interpR2 = si.PchipInterpolator(logR2, P2)
    rmin, rmax = max(min(logR1), min(logR2)), min(max(logR1), max(logR2))
    bd_psnr = quad(lambda r: interpR1(r)-interpR2(r), rmin, rmax)[0] / (rmax-rmin)

    return bd_rate, bd_psnr

# --- Main evaluation ---
# def evaluate_dataset(image_dir, metric="ms-ssim", qualities=range(1,9), out_prefix="bmshj2018", max_images=30):
#     results = []
#     # List all image files
#     image_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir) 
#                    if f.lower().endswith((".png",".jpg",".jpeg"))]
#     # Keep only the first `max_images` files
#     image_files = image_files[:max_images]

#     for q in qualities:
#         print(f"Testing quality={q}, metric={metric} on {len(image_files)} images...")
#         model = bmshj2018_hyperprior(quality=q, pretrained=True, metric=metric).to(device).eval()

#         bpps, psnrs = [], []
#         for path in image_files:
#             x = load_image(path)
#             bpp, P = test_model(model, x)
#             bpps.append(bpp)
#             psnrs.append(P)

#         mean_bpp = np.mean(bpps)
#         mean_psnr = np.mean(psnrs)
#         results.append((q, mean_bpp, mean_psnr))

#     df = pd.DataFrame(results, columns=["quality","bpp","psnr"])
#     df.to_csv(f"{out_prefix}_{metric}_results_msssim.csv", index=False)

#     # Plot RD curve
#     plt.figure()
#     plt.plot(df["bpp"], df["psnr"], "o-", label=f"{out_prefix} ({metric})")
#     plt.xlabel("Bitrate (bpp)")
#     plt.ylabel("PSNR (dB)")
#     plt.title("RD curve")
#     plt.legend()
#     plt.grid(True)
#     # plt.savefig(f"{out_prefix}_{metric}_rd_curve.png")
#     save_folder = "/dcs/large/u2157170/code/modelv2"
#     os.makedirs(save_folder, exist_ok=True)
#     plt.savefig(os.path.join(save_folder, f"{out_prefix}_{metric}_rd_curve_msssim.png"))
#     plt.close()

#     return df


# # Example usage
# if __name__ == "__main__":
#     dataset_folder = "/dcs/large/u2157170/code/results"
#     df = evaluate_dataset(dataset_folder, metric="ms-ssim")  # average PSNR/bpp per quality
#     print(df)

def evaluate_dataset_msssim(image_dir, qualities=range(1,9), out_prefix="bmshj2018", max_images=30):
    results = []
    image_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir) 
                   if f.lower().endswith((".png",".jpg",".jpeg"))][:max_images]

    for q in qualities:
        print(f"Testing quality={q} on {len(image_files)} images...")
        model = bmshj2018_hyperprior(quality=q, pretrained=True, metric="mse").to(device).eval()

        bpps, msssim_vals = [], []
        for path in image_files:
            x = load_image(path)
            bpp, ms_val = test_model_msssim(model, x)
            bpps.append(bpp)
            msssim_vals.append(ms_val)

        results.append((q, np.mean(bpps), np.mean(msssim_vals)))

    df = pd.DataFrame(results, columns=["quality","bpp","ms-ssim"])
    df.to_csv(f"{out_prefix}_msssim_results.csv", index=False)

    # Plot RD curve
    plt.figure()
    plt.plot(df["bpp"], df["ms-ssim"], "o-", label=f"{out_prefix} (MS-SSIM)")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("MS-SSIM")
    plt.title("RD curve")
    plt.legend()
    plt.grid(True)

    save_folder = "/dcs/large/u2157170/code/modelv2"
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{out_prefix}_msssim_rd_curve.png"))
    plt.close()

    return df


if __name__ == "__main__":
    dataset_folder = "/dcs/large/u2157170/code/results"
    df_msssim = evaluate_dataset_msssim(dataset_folder)
    print(df_msssim)
