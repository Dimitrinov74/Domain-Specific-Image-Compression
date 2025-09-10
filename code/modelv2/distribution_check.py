import torch
from model import CompressionModel
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, laplace, t

# --- load model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CompressionModel(N=128, M=192, spatial_params=False, min_nu=2.0, max_nu=100.0).to(device)
state = torch.load("/dcs/large/u2157170/code/modelv2/checkpoints/default_run/final_l_bigdata.pt", map_location=device)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

# --- load image ---
img = Image.open("/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_00_RGB.png").convert("RGB")
x = TF.to_tensor(img).unsqueeze(0).to(device)

# --- forward pass ---
with torch.no_grad():
    out = model(x, quant_mode="round")

y_tilde = out["y_tilde"].cpu().numpy()[0]  # shape [C, H, W]
num_channels = y_tilde.shape[0]

# Store results for all channels
channel_bestfits = {}
studentt_params = {}  # store params for Student-t fits

for ch in range(num_channels):
    data = y_tilde[ch].flatten()

    results = {}
    try:
        mu, std = norm.fit(data)
        results["Gaussian"] = np.sum(norm.logpdf(data, mu, std))
    except Exception:
        results["Gaussian"] = float("-inf")

    try:
        loc_l, scale_l = laplace.fit(data)
        results["Laplacian"] = np.sum(laplace.logpdf(data, loc_l, scale_l))
    except Exception:
        results["Laplacian"] = float("-inf")

    try:
        df, loc_t, scale_t = t.fit(data)
        loglik = np.sum(t.logpdf(data, df, loc_t, scale_t))
        results["StudentT"] = loglik
        studentt_params[ch] = (df, loc_t, scale_t)  # save params
    except Exception:
        results["StudentT"] = float("-inf")

    best = max(results.items(), key=lambda kv: kv[1])[0]
    channel_bestfits[ch] = best

# --- pick the 9 channels where StudentT is the best fit ---
student_t_channels = [ch for ch, best in channel_bestfits.items() if best == "StudentT"]
print(f"Found {len(student_t_channels)} channels best fit by StudentT.")

if len(student_t_channels) >= 9:
    selected_channels = student_t_channels[:9]
else:
    selected_channels = student_t_channels  # fewer than 9 available

# --- plot the 9 feature maps ---
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, ch in enumerate(selected_channels):
    ax = axes[i // 3, i % 3]
    fmap = y_tilde[ch]
    vmin, vmax = np.percentile(fmap, 1), np.percentile(fmap, 99)
    ax.imshow(fmap, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Channel {ch} (StudentT)")
    ax.axis("off")
for j in range(len(selected_channels), 9):
    axes[j // 3, j % 3].axis("off")
plt.tight_layout()
plt.savefig("/dcs/large/u2157170/code/modelv2/studentt_best_featuremaps.png")
print("Saved 9 StudentT-best feature maps to /dcs/large/u2157170/code/modelv2/studentt_best_featuremaps.png")

# --- plot histograms + Student-t fits ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, ch in enumerate(selected_channels):
    ax = axes[i // 3, i % 3]
    data = y_tilde[ch].flatten()
    ax.hist(data, bins=80, density=True, alpha=0.5, color="gray", label="Histogram")
    if ch in studentt_params:
        df, loc_t, scale_t = studentt_params[ch]
        x_axis = np.linspace(np.min(data), np.max(data), 500)
        ax.plot(x_axis, t.pdf(x_axis, df, loc_t, scale_t), "m-", lw=2, label=f"StudentT ν={df:.1f}")
    ax.set_title(f"Channel {ch} Histogram")
    ax.legend(fontsize=8)
for j in range(len(selected_channels), 9):
    axes[j // 3, j % 3].axis("off")
plt.tight_layout()
plt.savefig("/dcs/large/u2157170/code/modelv2/studentt_best_histograms.png")
print("Saved 9 StudentT-best histograms to /dcs/large/u2157170/code/modelv2/studentt_best_histograms.png")



