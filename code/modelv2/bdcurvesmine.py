import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from model import CompressionModel, rate_distortion_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- helpers ---
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

def load_image(path):
    img = Image.open(path).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(device)
    return x

def test_model(model, x, lambda_rd=1.0, dist="msssim"):
    """
    Evaluate a single image with the model and return (bpp, ms-ssim)
    """
    x_padded, pad_h, pad_w = pad_to_multiple(x)
    with torch.no_grad():
        out = model(x_padded, quant_mode="round")

    x_hat = out["x_hat"]
    x_hat = unpad(x_hat, pad_h, pad_w)[:, :, :x.size(2), :x.size(3)]
    out["x_hat"] = x_hat

    # Compute R and D using rate_distortion_loss
    loss, R, D = rate_distortion_loss(out, x, lambda_rd=lambda_rd, dist=dist)

    if dist.lower() in ["ms-ssim", "msssim"]:
        ms_ssim_val = 1.0 - D.item()  # D is 1 - MS-SSIM
        return R.item(), ms_ssim_val
    else:
        # fallback to MSE-based RD
        psnr_val = 10 * torch.log10(1.0 / D)
        return R.item(), psnr_val.item()
    
def test_model_psnr(model, x, lambda_rd=1.0, dist="mse"):
    """
    Evaluate a single image with the model and return (bpp, psnr)
    """
    x_padded, pad_h, pad_w = pad_to_multiple(x)
    with torch.no_grad():
        out = model(x_padded, quant_mode="round")

    x_hat = out["x_hat"]
    x_hat = unpad(x_hat, pad_h, pad_w)[:, :, :x.size(2), :x.size(3)]
    out["x_hat"] = x_hat

    # Compute R and D using rate_distortion_loss
    loss, R, D = rate_distortion_loss(out, x, lambda_rd=lambda_rd, dist=dist)

    # Compute PSNR from MSE
    mse = D.item()
    psnr_val = 10 * np.log10(1.0 / mse)
    return R.item(), psnr_val

def test_model_mse(model, x, lambda_rd=1.0, dist="mse"):
    """
    Evaluate a single image with the model and return (bpp, mse)
    """
    x_padded, pad_h, pad_w = pad_to_multiple(x)
    with torch.no_grad():
        out = model(x_padded, quant_mode="round")

    x_hat = out["x_hat"]
    x_hat = unpad(x_hat, pad_h, pad_w)[:, :, :x.size(2), :x.size(3)]
    out["x_hat"] = x_hat

    # Compute R and D using rate_distortion_loss
    loss, R, D = rate_distortion_loss(out, x, lambda_rd=lambda_rd, dist=dist)

    mse_val = D.item()  # D is already MSE
    return R.item(), mse_val

# --- evaluation ---
# def evaluate_multiple_checkpoints(image_dir, ckpt_folder, lambdas=[10,25,50,100,250,500,1000,10000],
#                                   max_images=30, out_prefix="custom_model"):

#     image_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir)
#                    if f.lower().endswith((".png",".jpg",".jpeg"))][:max_images]

#     results = []

#     for lam in lambdas:
#         ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")  # adjust naming
#         print(f"\nLoading checkpoint for lambda={lam}: {ckpt_path}")

#         # Load model
#         N, M = 128, 192
#         spatial_params = False
#         min_nu, max_nu = 2, 100
#         model = CompressionModel(N=N, M=M, spatial_params=spatial_params,
#                                  min_nu=min_nu, max_nu=max_nu).to(device)
#         state = torch.load(ckpt_path, map_location=device)
#         if "model" in state:
#             model.load_state_dict(state["model"])
#         else:
#             model.load_state_dict(state)
#         model.eval()

#         bpps, msssim_vals = [], []
#         print(f"Evaluating {len(image_files)} images...")
#         for path in image_files:
#             x = load_image(path)
#             bpp, ms_ssim_val = test_model(model, x, dist="msssim")
#             bpps.append(bpp)
#             msssim_vals.append(ms_ssim_val)

#         results.append((lam, np.mean(bpps), np.mean(msssim_vals)))

#     # Save CSV
#     df = pd.DataFrame(results, columns=["lambda","bpp","ms-ssim"])
#     csv_path = os.path.join(image_dir, f"{out_prefix}_ms-ssim_results_all_ckpts.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\nSaved CSV to {csv_path}")

#     # Plot RD curve
#     plt.figure()
#     plt.plot(df["bpp"], df["ms-ssim"], "o-", label=f"{out_prefix} (MS-SSIM)")
#     plt.xlabel("Bitrate (bpp)")
#     plt.ylabel("MS-SSIM")
#     plt.title("RD curve")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(image_dir, f"{out_prefix}_ms-ssim_rd_curve_all_ckpts.png"))
#     plt.close()
#     print(f"Saved RD curve to {image_dir}")

#     return df



# # --- usage ---
# if __name__ == "__main__":
#     image_dir = "/dcs/large/u2157170/code/results"
#     ckpt_folder = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
#     lambdas = [10,25,50,100,250,500,1000,10000]
#     df = evaluate_multiple_checkpoints(image_dir, ckpt_folder, lambdas)
#     print(df)

##############################################################################################
#############################################################################################

# def evaluate_multiple_checkpoints_psnr(image_dir, ckpt_folder, lambdas=[10,25,50,100,250,500,1000,10000],
#                                         max_images=30, out_prefix="custom_model"):

#     image_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir)
#                    if f.lower().endswith((".png",".jpg",".jpeg"))][:max_images]

#     results = []

#     for lam in lambdas:
#         ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")
#         print(f"\nLoading checkpoint for lambda={lam}: {ckpt_path}")

#         # Load model
#         N, M = 128, 192
#         spatial_params = False
#         min_nu, max_nu = 2, 100
#         model = CompressionModel(N=N, M=M, spatial_params=spatial_params,
#                                  min_nu=min_nu, max_nu=max_nu).to(device)
#         state = torch.load(ckpt_path, map_location=device)
#         if "model" in state:
#             model.load_state_dict(state["model"])
#         else:
#             model.load_state_dict(state)
#         model.eval()

#         bpps, psnr_vals = [], []
#         print(f"Evaluating {len(image_files)} images...")
#         for path in image_files:
#             x = load_image(path)
#             bpp, psnr_val = test_model_psnr(model, x, dist="mse")
#             bpps.append(bpp)
#             psnr_vals.append(psnr_val)

#         results.append((lam, np.mean(bpps), np.mean(psnr_vals)))

#     # Save CSV
#     df = pd.DataFrame(results, columns=["lambda","bpp","psnr"])
#     csv_path = os.path.join(image_dir, f"{out_prefix}_psnr_results_all_ckpts.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\nSaved CSV to {csv_path}")

#     # Plot RD curve
#     plt.figure()
#     plt.plot(df["bpp"], df["psnr"], "o-", label=f"{out_prefix} (PSNR)")
#     plt.xlabel("Bitrate (bpp)")
#     plt.ylabel("PSNR (dB)")
#     plt.title("RD curve")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(image_dir, f"{out_prefix}_psnr_rd_curve_all_ckpts.png"))
#     plt.close()
#     print(f"Saved RD curve to {image_dir}")

#     return df

# # Usage
# if __name__ == "__main__":
#     image_dir = "/dcs/large/u2157170/code/results"
#     ckpt_folder = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
#     lambdas = [10,25,50,100,250,500,1000,10000]
#     df_psnr = evaluate_multiple_checkpoints_psnr(image_dir, ckpt_folder, lambdas)
#     print(df_psnr)

##############################################################################################
#############################################################################################

def evaluate_multiple_checkpoints_mse(image_dir, ckpt_folder, lambdas=[10,25,50,100,250,500,1000,10000],
                                      max_images=30, out_prefix="custom_model"):

    image_files = [os.path.join(image_dir,f) for f in os.listdir(image_dir)
                   if f.lower().endswith((".png",".jpg",".jpeg"))][:max_images]

    results = []

    for lam in lambdas:
        ckpt_path = os.path.join(ckpt_folder, f"alpha{lam}lambda.pt")
        print(f"\nLoading checkpoint for lambda={lam}: {ckpt_path}")

        # Load model
        N, M = 128, 192
        spatial_params = False
        min_nu, max_nu = 2, 100
        model = CompressionModel(N=N, M=M, spatial_params=spatial_params,
                                 min_nu=min_nu, max_nu=max_nu).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        model.eval()

        bpps, mse_vals = [], []
        print(f"Evaluating {len(image_files)} images...")
        for path in image_files:
            x = load_image(path)
            bpp, mse_val = test_model_mse(model, x, dist="mse")
            bpps.append(bpp)
            mse_vals.append(mse_val)

        results.append((lam, np.mean(bpps), np.mean(mse_vals)))

    # Save CSV
    df = pd.DataFrame(results, columns=["lambda","bpp","mse"])
    csv_path = os.path.join(image_dir, f"{out_prefix}_mse_results_all_ckpts.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # Plot RD curve
    plt.figure()
    plt.plot(df["bpp"], df["mse"], "o-", label=f"{out_prefix} (MSE)")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("MSE")
    plt.title("RD curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(image_dir, f"{out_prefix}_mse_rd_curve_all_ckpts.png"))
    plt.close()
    print(f"Saved RD curve to {image_dir}")

    return df

# Usage
if __name__ == "__main__":
    image_dir = "/dcs/large/u2157170/code/results"
    ckpt_folder = "/dcs/large/u2157170/code/modelv2/checkpoints/default_run"
    lambdas = [10,25,50,100,250,500,1000,10000]
    df_mse = evaluate_multiple_checkpoints_mse(image_dir, ckpt_folder, lambdas)
    print(df_mse)
