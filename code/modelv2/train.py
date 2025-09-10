
# import os, argparse, random
# import importlib.util
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from rich import print

# from datasets import make_loaders
# from model import CompressionModel, rate_distortion_loss

# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def load_config(path):
#     spec = importlib.util.spec_from_file_location("cfg", path)
#     cfg = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(cfg)
#     return cfg

# def save_ckpt(state, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(state, path)

# def validate(model, loader, cfg, device):
#     model.eval()
#     tot_R, tot_D, n = 0.0, 0.0, 0
#     with torch.no_grad():
#         for x in loader:
#             x = x.to(device)
#             out = model(x, quant_mode="round")  # round at val
#             loss, R, D = rate_distortion_loss(out, x, lambda_rd=cfg.LOSS.lambda_rd, dist=cfg.LOSS.dist)
#             b = x.size(0)
#             tot_R += R.item() * b
#             tot_D += D.item() * b
#             n += b
#     model.train()
#     return tot_R/n, tot_D/n

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config.py")
#     parser.add_argument("--run_name", type=str, default="default_run")
#     parser.add_argument("--epochs", type=int, default=None)
#     parser.add_argument("--lambda_rd", type=float, default=None)
#     parser.add_argument("--dist", type=str, default=None, choices=["mse","msssim"])
#     args = parser.parse_args()

#     cfg = load_config(args.config)
#     if args.epochs is not None: cfg.TRAIN.epochs = args.epochs
#     if args.lambda_rd is not None: cfg.LOSS.lambda_rd = args.lambda_rd
#     if args.dist is not None: cfg.LOSS.dist = args.dist

#     set_seed(cfg.TRAIN.seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     train_loader, val_loader = make_loaders(cfg)

#     model = CompressionModel(
#         N=cfg.MODEL.N, M=cfg.MODEL.M,
#         spatial_params=cfg.MODEL.spatial_params,
#         min_nu=cfg.MODEL.min_nu, max_nu=cfg.MODEL.max_nu
#     ).to(device)

#     opt = optim.Adam(model.parameters(), lr=cfg.OPTIM.lr, betas=cfg.OPTIM.betas, weight_decay=cfg.OPTIM.weight_decay)
#     scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.amp)

#     global_step = 0
#     best_val_R = float("inf")
#     ckpt_dir = os.path.join(cfg.LOGGING.ckpt_dir, args.run_name)
#     os.makedirs(ckpt_dir, exist_ok=True)

#     for epoch in range(1, cfg.TRAIN.epochs+1):
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.TRAIN.epochs}")
#         for i, x in enumerate(pbar, start=1):
#             x = x.to(device, non_blocking=True)
#             opt.zero_grad(set_to_none=True)
#             with torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp):
#                 out = model(x, quant_mode="noise")
#                 loss, R, D = rate_distortion_loss(out, x, lambda_rd=cfg.LOSS.lambda_rd, dist=cfg.LOSS.dist)
#             scaler.scale(loss).backward()
#             if cfg.OPTIM.grad_clip is not None and cfg.OPTIM.grad_clip > 0:
#                 scaler.unscale_(opt)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.grad_clip)
#             scaler.step(opt)
#             scaler.update()

#             global_step += 1
#             if global_step % cfg.LOGGING.log_every == 0:
#                 pbar.set_postfix({"loss": f"{loss.item():.4f}", "R(bpp)": f"{R.item():.3f}", "D": f"{D.item():.4f}"})

#             if global_step % cfg.LOGGING.val_every == 0:
#                 val_R, val_D = validate(model, val_loader, cfg, device)
#                 print(f"[step {global_step}] val R={val_R:.3f} bpp | val D={val_D:.5f}")
#                 is_best = val_R < best_val_R
#                 if is_best:
#                     best_val_R = val_R
#                 save_ckpt({
#                     "model": model.state_dict(),
#                     "opt": opt.state_dict(),
#                     "scaler": scaler.state_dict(),
#                     "epoch": epoch,
#                     "step": global_step,
#                     "best_val_R": best_val_R,
#                     "cfg": {k:getattr(cfg,k).__dict__ for k in ["DATA","OPTIM","MODEL","TRAIN","LOSS","LOGGING"]},
#                 }, os.path.join(ckpt_dir, f"step_{global_step:07d}{'_best' if is_best else ''}.pt"))

#     print("Training done. Saving final checkpoint...")
#     save_ckpt({
#         "model": model.state_dict(),
#         "opt": opt.state_dict(),
#         "scaler": scaler.state_dict(),
#         "epoch": epoch,
#         "step": global_step,
#         "best_val_R": best_val_R,
#     }, os.path.join(ckpt_dir, f"final.pt"))

# if __name__ == "__main__":
#     main()

import os, random, argparse
import importlib.util
import torch
import torch.optim as optim
from tqdm import tqdm
from rich import print

from datasets import make_loaders
from model import CompressionModel, rate_distortion_loss

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def validate(model, loader, cfg, device):
    model.eval()
    tot_R, tot_D, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out = model(x, quant_mode="round")
            loss, R, D = rate_distortion_loss(out, x, lambda_rd=cfg.LOSS.lambda_rd, dist="msssim")
            b = x.size(0)
            tot_R += R.item() * b
            tot_D += D.item() * b
            n += b
    model.train()
    return tot_R/n, tot_D/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.py")
    parser.add_argument("--run_name", type=str, default="default_run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.TRAIN.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = make_loaders(cfg)

    model = CompressionModel(
        N=cfg.MODEL.N, M=cfg.MODEL.M,
        spatial_params=cfg.MODEL.spatial_params,
        min_nu=cfg.MODEL.min_nu, max_nu=cfg.MODEL.max_nu
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.OPTIM.lr,
                     betas=cfg.OPTIM.betas, weight_decay=cfg.OPTIM.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.amp)

    global_step = 0
    best_val_R = float("inf")
    ckpt_dir = os.path.join(cfg.LOGGING.ckpt_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.TRAIN.epochs+1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.TRAIN.epochs}")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.amp):
                out = model(x, quant_mode="noise")
                loss, R, D = rate_distortion_loss(out, x, lambda_rd=cfg.LOSS.lambda_rd, dist="msssim")
            scaler.scale(loss).backward()
            if cfg.OPTIM.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % cfg.LOGGING.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "R(bpp)": f"{R.item():.3f}",
                                  "MS-SSIM": f"{1-D.item():.4f}"})

            if global_step % cfg.LOGGING.val_every == 0:
                val_R, val_D = validate(model, val_loader, cfg, device)
                print(f"[step {global_step}] val R={val_R:.3f} bpp | val (1-MS-SSIM)={val_D:.5f}")
                # is_best = val_R < best_val_R
                # if is_best:
                #     best_val_R = val_R
                # save_ckpt({
                #     "model": model.state_dict(),
                #     "opt": opt.state_dict(),
                #     "scaler": scaler.state_dict(),
                #     "epoch": epoch,
                #     "step": global_step,
                #     "best_val_R": best_val_R,
                # }, os.path.join(ckpt_dir, f"step_{global_step:07d}{'_best' if is_best else ''}.pt"))

    print("Training done. Saving final checkpoint...")
    save_ckpt({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "best_val_R": best_val_R,
    }, os.path.join(ckpt_dir, f"alpha10000lambda.pt"))

if __name__ == "__main__":
    main()

