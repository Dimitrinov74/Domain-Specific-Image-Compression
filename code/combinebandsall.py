import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Normalize to [0, 1]
def normalize(band):
    band = band.astype(np.float32)
    band -= band.min()
    if band.max() != 0:
        band /= band.max()
    return band

# Create RGB image from a single patch
def process_patch(patch_path, output_dir):
    # Look for .tif files for bands
    files = os.listdir(patch_path)
    b2 = next((f for f in files if '_B02' in f), None)
    b3 = next((f for f in files if '_B03' in f), None)
    b4 = next((f for f in files if '_B04' in f), None)

    if not (b2 and b3 and b4):
        print(f"Missing bands in {patch_path}")
        return

    # Read each band
    with rasterio.open(os.path.join(patch_path, b2)) as src:
        blue = normalize(src.read(1))
    with rasterio.open(os.path.join(patch_path, b3)) as src:
        green = normalize(src.read(1))
    with rasterio.open(os.path.join(patch_path, b4)) as src:
        red = normalize(src.read(1))

    # Stack into RGB
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_img = (rgb * 255).astype(np.uint8)

    # Output filename
    patch_name = os.path.basename(patch_path.rstrip('/'))
    output_path = os.path.join(output_dir, f"{patch_name}_RGB.png")

    # Save using matplotlib
    plt.imsave(output_path, rgb_img)
    print(f"Saved: {output_path}")

# Main: loop through all patch folders
def process_all_patches(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.listdir(root_dir):
        patch_path = os.path.join(root_dir, folder)
        if os.path.isdir(patch_path):
            process_patch(patch_path, output_dir)

# ======== USAGE ==========
# Set these paths
PATCHES_ROOT = "/dcs/large/u2157170/testfold3/BigEarthNet-S2/S2B_MSIL2A_20170825T093029_N9999_R136_T34TEQ"
OUTPUT_DIR = "/dcs/large/u2157170/code/results"

process_all_patches(PATCHES_ROOT, OUTPUT_DIR)