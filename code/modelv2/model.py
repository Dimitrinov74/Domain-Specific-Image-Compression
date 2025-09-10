
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import piq
import config

from layers import AnalysisTransform, SynthesisTransform, HyperAnalysis, HyperSynthesis
from distributions import StudentT, FactorizedGaussian

class CompressionModel(nn.Module):
    def __init__(self, N=128, M=192, spatial_params=False, min_nu=1.1, max_nu=100.0):
        super().__init__()
        self.g_a = AnalysisTransform(N, M)
        self.g_s = SynthesisTransform(N, M)
        self.h_a = HyperAnalysis(M, N)
        self.h_s = HyperSynthesis(N, M, spatial_params=spatial_params)

        self.studentT = StudentT()
        self.z_prior = FactorizedGaussian(N)

        self.min_nu = min_nu
        self.max_nu = max_nu
        self.spatial_params = spatial_params

    @staticmethod
    def quantize(x, mode):
        if mode == "noise":
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            return x + noise
        elif mode == "round":
            return torch.round(x)
        else:
            raise ValueError(f"Unknown quant mode: {mode}")

    def forward(self, x, quant_mode="noise"):
        # Analysis
        y = self.g_a(x)
        # Hyperanalysis
        z = self.h_a(y)

        # Relaxed quantization
        y_tilde = self.quantize(y, quant_mode)
        z_tilde = self.quantize(z, quant_mode)

        # Hyperdecoder predicts parameters for Student-t over y
        log_sigma, log_nu = self.h_s(z_tilde)
        if self.spatial_params:
            sigma = torch.exp(log_sigma)
            nu = torch.clamp(torch.exp(log_nu), min=self.min_nu, max=self.max_nu)
        else:
            # global per-channel: shape [N,M,1,1] -> broadcast to y shape
            sigma = torch.exp(log_sigma).mean(dim=(2,3), keepdim=True).expand_as(y_tilde)
            nu = torch.clamp(torch.exp(log_nu).mean(dim=(2,3), keepdim=True), self.min_nu, self.max_nu).expand_as(y_tilde)

        # Likelihoods (bits per element)
        nll_y = self.studentT.neg_log2_prob(y_tilde, sigma=sigma, nu=nu)
        nll_z = self.z_prior.neg_log2_prob(z_tilde)

        # Synthesis
        y_hat = self.quantize(y, "round") if self.training is False else y_tilde
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "nll_y": nll_y,
            "nll_z": nll_z,
            "y": y, "y_tilde": y_tilde,
            "z": z, "z_tilde": z_tilde,
            "sigma": sigma, "nu": nu,
        }


def rate_distortion_loss(out, x, lambda_rd=10000.0, dist="mssim"):
    N, C, H, W = x.shape
    # R = (out["nll_y"].sum() + out["nll_z"].sum()) / (N * H * W)
    R = (out["nll_y"].sum() + out["nll_z"].sum()) / (N * H * W)
    R = torch.clamp(R, min=0.0)

    if dist == "mse":
        D = F.mse_loss(out["x_hat"], x)
    # elif dist == "msssim":
    #     x_hat = out["x_hat"]
    #     device = x.device
    #     scale_weights = torch.tensor([0.3, 0.5, 0.2], device=device)
    #     D = 1.0 - piq.multi_scale_ssim(
    #         x_hat.clamp(0, 1),
    #         x,
    #         data_range=1.0,
    #         scale_weights=scale_weights
    #     )
    elif dist == "msssim":
        x_hat = out["x_hat"]
        if x_hat.shape[2:] != x.shape[2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[2:], mode="bilinear", align_corners=False)
        D = 1.0 - piq.multi_scale_ssim(
            x_hat.clamp(0, 1),
            x,
            data_range=1.0,
            scale_weights=torch.tensor([0.3, 0.5, 0.2], device=x.device)
        )
    else:
        raise ValueError("dist must be 'mse' or 'msssim'")

    loss = lambda_rd * D + R
    return loss, R.detach(), D.detach()






