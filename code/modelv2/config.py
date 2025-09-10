
from dataclasses import dataclass

@dataclass
class DATA:
    root: str = "/dcs/large/u2157170/code/results/"
    # val_root: str   = "/path/to/val_images"
    crop_size: int  = 256
    num_workers: int = 8
    batch_size: int = 16

@dataclass
class OPTIM:
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0

@dataclass
class MODEL:
    # Analysis/synthesis channels
    N: int = 128
    M: int = 192
    # Predict global (per-channel) params (False) or spatial maps (True)
    spatial_params: bool = False
    # Minimum/maximum degrees of freedom for Student-t
    min_nu: float = 2
    max_nu: float = 100.0

@dataclass
class TRAIN:
    seed: int = 42
    epochs: int = 30
    amp: bool = True

@dataclass
class LOSS:
    lambda_rd: float = 10000.0  # will be overridden by CLI if provided
    dist: str = "msssim"        # "mse" or "msssim"

@dataclass
class LOGGING:
    ckpt_dir: str = "checkpoints"
    log_every: int = 100
    val_every: int = 1000
