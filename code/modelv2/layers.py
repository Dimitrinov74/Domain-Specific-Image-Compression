
import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    """Generalized Divisive Normalization (simple, stable implementation)."""
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2**-18):
        super().__init__()
        self.inverse = inverse
        self.reparam_offset = reparam_offset
        self.beta = nn.Parameter(torch.sqrt(torch.ones(channels) + reparam_offset))
        self.gamma = nn.Parameter(torch.sqrt(torch.eye(channels) * gamma_init + reparam_offset))
        # Use depthwise 1x1 for gamma
        self.gamma_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        with torch.no_grad():
            self.gamma_conv.weight.copy_(self.gamma.diag().view(channels,1,1,1))

    def forward(self, x):
        beta = self.beta**2 - self.reparam_offset
        gamma = (self.gamma_conv.weight**2 - self.reparam_offset)
        # denom = sqrt(beta + gamma * x^2) depthwise
        denom = torch.sqrt(beta.view(1,-1,1,1) + F.conv2d(x**2, gamma, bias=None, groups=x.size(1)))
        if self.inverse:
            return x * denom
        else:
            return x / denom

def conv(in_ch, out_ch, k, stride=1):
    p = (k-1)//2
    return nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=p)

# class AnalysisTransform(nn.Module):
#     def __init__(self, N=128, M=192):
#         super().__init__()
#         self.g_a = nn.Sequential(
#             conv(3, N, 5, 2), GDN(N),
#             conv(N, N, 5, 2), GDN(N),
#             conv(N, N, 5, 2), GDN(N),
#             conv(N, M, 5, 2),
#         )

#     def forward(self, x):
#         return self.g_a(x)

class AnalysisTransform(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.g_a = nn.Sequential(

            conv(3, N, 3, 1), GDN(N),

            # Layer 1: downsample + GDN
            conv(N, N, 5, 2), GDN(N),
            # Extra conv for feature extraction (no downsampling)
            conv(N, N, 3, 1), GDN(N),


            # Layer 2: downsample + GDN
            conv(N, N, 5, 2), GDN(N),
            # Extra conv for feature extraction (no downsampling)
            conv(N, N, 3, 1), GDN(N),

            # Layer 3: downsample + GDN
            conv(N, N, 5, 2), GDN(N),

            conv(N, N, 3, 1), GDN(N),

            # conv(N, N, 3, 1), GDN(N),

            # Layer 4: downsample to latent space M
            conv(N, M, 5, 2),
        )

    def forward(self, x):
        return self.g_a(x)

class SynthesisTransform(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.g_s = nn.Sequential(
            # Layer 4: upsample from latent space M - N
            nn.ConvTranspose2d(M, N, 5, 2, 2, output_padding=1), GDN(N, inverse=True),

            # conv layer (no size change)
            conv(N, N, 3, 1), GDN(N, inverse=True),

            # Layer 3: upsample + IGDN
            nn.ConvTranspose2d(N, N, 5, 2, 2, output_padding=1), GDN(N, inverse=True),
            conv(N, N, 3, 1), GDN(N, inverse=True),

            # Layer 2: upsample + IGDN
            nn.ConvTranspose2d(N, N, 5, 2, 2, output_padding=1), GDN(N, inverse=True),
            conv(N, N, 3, 1), GDN(N, inverse=True),

            # Layer 1: upsample + IGDN
            nn.ConvTranspose2d(N, 3, 5, 2, 2, output_padding=1),
        )

    def forward(self, y_hat):
        return self.g_s(y_hat)


class HyperAnalysis(nn.Module):
    def __init__(self, M=192, N=128):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(M, N, 3, 1), nn.ReLU(inplace=True),
            conv(N, N, 3, 1), nn.ReLU(inplace=True),
            # conv(M, N, 3, 1), nn.ReLU(inplace=True),
            conv(N, N, 5, 2), nn.ReLU(inplace=True),
            conv(N, N, 5, 2),
        )

    def forward(self, y):
        return self.h_a(y)

class HyperSynthesis(nn.Module):
    def __init__(self, N=128, M=128, spatial_params=False):
        super().__init__()
        self.spatial_params = spatial_params
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, output_padding=1), nn.ReLU(inplace=True),
        )
        # Heads for log_sigma and log_nu (per-channel)
        if spatial_params:
            self.to_sigma = conv(N, M, 3, 1)
            self.to_nu    = conv(N, M, 3, 1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.mlp_sigma = nn.Sequential(
                nn.Conv2d(N, N, 1), nn.ReLU(),
                nn.Conv2d(N, M, 1)   # <-- outputs M channels now
            )
            self.mlp_nu = nn.Sequential(
                nn.Conv2d(N, N, 1), nn.ReLU(),
                nn.Conv2d(N, M, 1)   # <-- outputs M channels now
            )

    def forward(self, z):
        t = self.h_s(z)
        if self.spatial_params:
            log_sigma = self.to_sigma(t)
            log_nu = self.to_nu(t)
        else:
            p = self.pool(t)
            log_sigma = self.mlp_sigma(p)          # [B, M, 1, 1]
            log_sigma = log_sigma.expand(-1, -1, t.size(2), t.size(3))  # expand spatially
            log_nu    = self.mlp_nu(p)             # [B, M, 1, 1]
            log_nu    = log_nu.expand(-1, -1, t.size(2), t.size(3))
        return log_sigma, log_nu

