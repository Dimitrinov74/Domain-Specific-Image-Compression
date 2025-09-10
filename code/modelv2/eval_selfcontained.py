import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# import your model + loss from model.py
from model import CompressionModel, rate_distortion_loss


def compute_psnr(x, y, max_val=1.0):
    """Compute PSNR between two tensors in [0,1]."""
    mse = F.mse_loss(x, y)
    if mse == 0:
        return torch.tensor(float("inf"), device=x.device)
    return 20.0 * torch.log10(torch.tensor(max_val, device=x.device)) - 10.0 * torch.log10(mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/dcs/large/u2157170/code/modelv2/checkpoints/default_run/alpha100lambda.pt",
                        help="Path to checkpoint file (.pt)")
    parser.add_argument("--image", type=str,
                        default="/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_02_RGB.png",
                        help="Path to input image (.png/.jpg)")
    parser.add_argument("--output", type=str,
                        default="/dcs/large/u2157170/code/modelv2/reconstruction_100lambda.png",
                        help="Path to save reconstruction image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- define model hyperparameters here ---
    N = 128
    M = 192
    spatial_params = False
    min_nu = 2
    max_nu = 100.0
    lambda_rd = 30.0
    dist = "msssim"   # or "mse"
    # -----------------------------------------

    # build model
    model = CompressionModel(N=N, M=M,
                             spatial_params=spatial_params,
                             min_nu=min_nu, max_nu=max_nu).to(device)
    model.eval()

    # load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    # load input image
    img = Image.open(args.image).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        out = model(x, quant_mode="round")
        x_hat = out["x_hat"]

        # crop reconstruction to match original
        x_hat = x_hat[:, :, :x.size(2), :x.size(3)]
        out["x_hat"] = x_hat

        loss, R, D = rate_distortion_loss(
            out,
            x,
            lambda_rd=lambda_rd,
            dist=dist
        )

    # --- metrics ---
    mse_val = F.mse_loss(x_hat, x).item()
    psnr_val = compute_psnr(x_hat, x).item()

    print("Original:", x.shape)
    print("Reconstruction:", x_hat.shape)
    print(f"Image bpp: {R.item():.3f}")
    print(f"Distortion term ({dist}): {D.item():.5f}")
    print(f"MSE: {mse_val:.6f}")
    print(f"PSNR: {psnr_val:.2f} dB")

    # save reconstruction
    rec = TF.to_pil_image(x_hat.squeeze(0).clamp(0, 1).cpu())
    rec.save(args.output)
    print(f"Saved reconstruction to {args.output}")


if __name__ == "__main__":
    main()

