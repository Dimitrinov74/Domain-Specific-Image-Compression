import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from compressai.zoo import bmshj2018_factorized

model = bmshj2018_factorized(quality=3, pretrained=True).eval()

analysis_transform = model.g_a

transform = T.Compose([
    T.ToTensor(),
    T.Pad((0,0,8,8))
])

def get_first_images(folder, count=5, extensions={'.png'}):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in extensions
    ])[:count]

def process_images(folder):
    image_paths = get_first_images(folder)

    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            y = analysis_transform(x)

        print(f"Image {idx+1}: {os.path.basename(path)}")
        print(f" -> Padded shape: {tuple(x.shape)}")
        print(f" -> Padded shape: {tuple(y.shape)}")
        print()

image_folder = "/dcs/large/u2157170/code/results/"
process_images(image_folder)