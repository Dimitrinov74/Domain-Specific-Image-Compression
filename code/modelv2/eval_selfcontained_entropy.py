#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchac
from pytorch_msssim import ms_ssim
from PIL import Image
import torchvision.transforms.functional as TF

from model import CompressionModel, rate_distortion_loss

# ---------------- helpers: CDF/PMF building ----------------
def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def pmf_to_uint16_cdf(pmf):
    cdf = torch.cumsum(pmf, dim=0)
    zero = torch.zeros(1, *cdf.shape[1:], device=cdf.device, dtype=cdf.dtype)
    cdf_with_zero = torch.cat([zero, cdf], dim=0)
    cdf_with_zero[-1].clamp_(min=1.0)
    cdf_scaled = (cdf_with_zero * 65535.0).clamp(0, 65535.0).cpu().numpy().astype(np.uint16)
    return cdf_scaled

# ---------------- compression / decompression ----------------
def custom_compress(model, x, tail=10):
    device = x.device
    B = x.size(0)
    out = model(x, quant_mode="round")
    y_q, z_q = out["y_tilde"].detach(), out["z_tilde"].detach()
    sigma_y, nu_y = out["sigma"].detach(), out["nu"].detach()
    sigma_z = torch.exp(model.z_prior.log_sigma).view(1, -1, 1, 1).to(device)

    strings, miny_list, maxy_list, minz_list, maxz_list = [], [], [], [], []

    for b in range(B):
        # z encoding
        zvals = z_q[b]
        zmin = int(torch.floor(zvals.min().item())) - tail
        zmax = int(torch.ceil(zvals.max().item())) + tail
        support_z = torch.arange(zmin, zmax + 1, device=device, dtype=torch.float32)
        Lz = support_z.numel()
        upper, lower = (support_z + 0.5).view(Lz, 1, 1, 1), (support_z - 0.5).view(Lz, 1, 1, 1)
        Fu, Fl = gaussian_cdf(upper / sigma_z), gaussian_cdf(lower / sigma_z)
        pmf_z = (Fu - Fl).clamp(min=1e-12)
        pmf_z = pmf_z / pmf_z.sum(dim=0, keepdim=True)
        cdf_z = pmf_to_uint16_cdf(pmf_z)
        zs_bytes = torchac.encode_float_cdf(cdf_z, (zvals.cpu().numpy().astype(np.int32) - zmin).astype(np.int32))

        # y encoding
        yvals = y_q[b]
        ymin = int(torch.floor(yvals.min().item())) - tail
        ymax = int(torch.ceil(yvals.max().item())) + tail
        support_y = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)
        Ly = support_y.numel()
        upper, lower = (support_y + 0.5).view(Ly, 1, 1, 1), (support_y - 0.5).view(Ly, 1, 1, 1)
        dist = torch.distributions.StudentT(df=nu_y[b], loc=torch.zeros_like(nu_y[b]), scale=sigma_y[b])
        Fu_y, Fl_y = dist.cdf(upper), dist.cdf(lower)
        pmf_y = (Fu_y - Fl_y).clamp(min=1e-12)
        pmf_y = pmf_y / pmf_y.sum(dim=0, keepdim=True)
        cdf_y = pmf_to_uint16_cdf(pmf_y)
        ys_bytes = torchac.encode_float_cdf(cdf_y, (yvals.cpu().numpy().astype(np.int32) - ymin).astype(np.int32))

        strings.append([zs_bytes, ys_bytes])
        miny_list.append(ymin); maxy_list.append(ymax)
        minz_list.append(zmin); maxz_list.append(zmax)

    return {
        "strings": strings,
        "shape_y": list(y_q.shape),
        "shape_z": list(z_q.shape),
        "min_y": miny_list, "max_y": maxy_list,
        "min_z": minz_list, "max_z": maxz_list,
    }

def custom_decompress(model, compressed):
    device = next(model.parameters()).device
    strings = compressed["strings"]
    shape_y, shape_z = compressed["shape_y"], compressed["shape_z"]
    miny_list, maxy_list = compressed["min_y"], compressed["max_y"]
    minz_list, maxz_list = compressed["min_z"], compressed["max_z"]

    recon_list = []
    for b in range(len(strings)):
        zs_bytes, ys_bytes = strings[b]
        zmin, zmax, ymin, ymax = minz_list[b], maxz_list[b], miny_list[b], maxy_list[b]

        sigma_z = torch.exp(model.z_prior.log_sigma).view(-1,1,1).to(device)
        support_z = torch.arange(zmin, zmax + 1, device=device, dtype=torch.float32)
        Lz = support_z.numel()
        Fu, Fl = gaussian_cdf((support_z + 0.5).view(Lz,1,1,1)/sigma_z), gaussian_cdf((support_z - 0.5).view(Lz,1,1,1)/sigma_z)
        pmf_z = (Fu - Fl).clamp(min=1e-12)
        pmf_z = pmf_z / pmf_z.sum(dim=0, keepdim=True)
        cdf_z = pmf_to_uint16_cdf(pmf_z)

        z_symbols = torchac.decode_float_cdf(cdf_z, zs_bytes, out_shape=(shape_z[1], shape_z[2], shape_z[3]))
        z_hat = torch.from_numpy(z_symbols + zmin).to(device).float().unsqueeze(0)

        with torch.no_grad():
            log_sigma_y, log_nu_y = model.h_s(z_hat)
            if model.spatial_params:
                sigma_y = torch.exp(log_sigma_y)
                nu_y = torch.clamp(torch.exp(log_nu_y), min=model.min_nu, max=model.max_nu)
            else:
                sigma_y = torch.exp(log_sigma_y).mean(dim=(2,3), keepdim=True).expand(-1,-1,shape_y[2],shape_y[3])
                nu_y = torch.clamp(torch.exp(log_nu_y).mean(dim=(2,3), keepdim=True), model.min_nu, model.max_nu).expand_as(sigma_y)

        support_y = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)
        Ly = support_y.numel()
        dist_y = torch.distributions.StudentT(df=nu_y.squeeze(0), loc=torch.zeros_like(nu_y.squeeze(0)), scale=sigma_y.squeeze(0))
        Fu_y, Fl_y = dist_y.cdf((support_y + 0.5).view(Ly,1,1,1)), dist_y.cdf((support_y - 0.5).view(Ly,1,1,1))
        pmf_y = (Fu_y - Fl_y).clamp(min=1e-12)
        pmf_y = pmf_y / pmf_y.sum(dim=0, keepdim=True)
        cdf_y = pmf_to_uint16_cdf(pmf_y)

        y_symbols = torchac.decode_float_cdf(cdf_y, ys_bytes, out_shape=(shape_y[1], shape_y[2], shape_y[3]))
        y_hat = torch.from_numpy(y_symbols + ymin).to(device).float().unsqueeze(0)

        with torch.no_grad():
            x_hat = model.g_s(y_hat)
        recon_list.append(x_hat)

    return torch.cat(recon_list, dim=0).clamp(0,1)

# ---------------- main evaluation function ----------------
def evaluate_image(ckpt_path, image_path, output_path="recon.png", tail=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CompressionModel(N=128, M=192, spatial_params=False, min_nu=2.0, max_nu=100.0).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(device)

    # forward pass for estimated bpp
    with torch.no_grad():
        out = model(x, quant_mode="round")
        loss, R_est, D = rate_distortion_loss(out, x, lambda_rd=1.0, dist="msssim")
    print(f"[forward] estimated bpp R = {R_est.item():.4f}, D = {D.item():.5f}")

    # compress
    compressed = custom_compress(model, x, tail=tail)
    total_bits = sum(len(s) * 8 for entry in compressed["strings"] for s in entry)
    bpp_real = total_bits / (x.size(-2) * x.size(-1))
    print(f"[entropy coding] compressed (real) bpp = {bpp_real:.4f}")

    # decompress
    x_hat = custom_decompress(model, compressed)
    mss = ms_ssim(x_hat, x, data_range=1.0, size_average=True).item()
    print(f"[entropy coding] MS-SSIM = {mss:.6f}")

    TF.to_pil_image(x_hat.squeeze(0).cpu()).save(output_path)
    print("Saved decompressed reconstruction to", output_path)
    return x_hat, compressed

# ---------------- example usage ----------------
if __name__ == "__main__":
    # Example: just call the function directly
    evaluate_image(
        ckpt_path="/dcs/large/u2157170/code/modelv2/checkpoints/default_run/final_l_bigdata.pt",
        image_path="/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_00_RGB.png",
        output_path="/dcs/large/u2157170/code/modelv2/reconstruction_l.png"
    )
