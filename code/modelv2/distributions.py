
import math
import torch
import torch.nn as nn

LOG2E = 1.0 / math.log(2.0)

def clamp_tensor(x, minv, maxv):
    return torch.clamp(x, min=minv, max=maxv)

class StudentT(nn.Module):
    """Per-channel Student-t with positive scale σ and dof ν.
    Expects inputs shaped [N, C, H, W]; parameters broadcastable to that.
    Returns -log2 p(x).
    """
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def neg_log2_prob(self, x, sigma, nu):
        # sigma = torch.clamp(sigma, min=self.eps)
        # nu = torch.clamp(nu, min=1.01)  # >1 for finite mean; >2 for finite var
        sigma = torch.clamp(sigma, min=1e-3, max=1e3)
        nu = torch.clamp(nu, min=2.0, max=100.0)
        # log pdf (natural log)
        # log C = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
        logC = torch.lgamma((nu+1.0)/2.0) - torch.lgamma(nu/2.0)                - 0.5*torch.log(nu*torch.pi) - torch.log(sigma)
        quad = (x / sigma)**2
        logp = logC - ((nu+1.0)/2.0) * torch.log1p(quad/nu)
        # return in bits
        return -logp * LOG2E

class FactorizedGaussian(nn.Module):
    """Simple zero-mean factorized Gaussian prior with learnable per-channel log_sigma."""
    def __init__(self, C):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(C))

    def neg_log2_prob(self, x):
        # x: [N,C,H,W]
        # sigma = torch.exp(self.log_sigma).view(1, -1, 1, 1)
        sigma = torch.exp(self.log_sigma).view(1, -1, 1, 1)
        sigma = torch.clamp(sigma, min=1e-3, max=1e3)
        var = sigma**2
        logp = -0.5*torch.log(2*torch.pi*var) - 0.5*(x**2)/var
        return -logp * LOG2E
