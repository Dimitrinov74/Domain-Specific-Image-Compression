from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_grayscale_png(path):
    img = Image.open(path).convert("L")
    return np.array(img)

def normalize(array):
    array = array.astype(np.float32)
    array -= array.min()
    if array.max() > 0:
        array /= array.max()
    return array

def create_rgb_from_pngs(b2_path, b3_path, b4_path, output_path):
    blue = normalize(load_grayscale_png(b2_path))
    green = normalize(load_grayscale_png(b3_path))
    red = normalize(load_grayscale_png(b4_path))

    rgb = np.stack([red, green, blue], axis = -1)

    rgb_unit8 = (rgb*255).astype(np.uint8)

    Image.fromarray(rgb_unit8).save(output_path)
    print("saved")

b2 = '/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_00_B02.tif.png'
b3 = '/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_00_B03.tif.png'
b4 = '/dcs/large/u2157170/code/results/S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_00_00_B04.tif.png'
output = '/dcs/large/u2157170/code/results/rgb_combined.png'

create_rgb_from_pngs(b2, b3, b4, output)