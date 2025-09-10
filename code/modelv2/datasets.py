import os, glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def list_images(root):
    exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff','*.webp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, '**', e), recursive=True))
    return files

class FolderImages(Dataset):
    def __init__(self, root_or_paths, crop_size, train=True):
        if isinstance(root_or_paths, (list, tuple)):
            self.paths = root_or_paths
        else:
            self.paths = list_images(root_or_paths)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found")
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    # def _pad_image(self, img):
    #     w, h = img.size
    #     pad_w = max(self.crop_size - w, 0)
    #     pad_h = max(self.crop_size - h, 0)
    #     if pad_w > 0 or pad_h > 0:
    #         # Pad equally on left/right and top/bottom
    #         pad_left = pad_w // 2
    #         pad_right = pad_w - pad_left
    #         pad_top = pad_h // 2
    #         pad_bottom = pad_h - pad_top
    #         img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='reflect')
    #     return img
    def _pad_to_multiple_of(self, img, mult=16):
        w, h = img.size
        new_w = ((w + mult - 1) // mult) * mult
        new_h = ((h + mult - 1) // mult) * mult
        pad_left = (new_w - w) // 2
        pad_top  = (new_h - h) // 2
        pad_right  = new_w - w - pad_left
        pad_bottom = new_h - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='reflect')

    def _pad_image(self, img):
        w, h = img.size
        min_size = max(self.crop_size, 161)  # ensure at least 161
        pad_w = max(min_size - w, 0)
        pad_h = max(min_size - h, 0)
        if pad_w > 0 or pad_h > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='reflect')
        return img


    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = self._pad_to_multiple_of(img)
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
        x = TF.to_tensor(img)  # [0,1]
        return x

def make_loaders(cfg):
    all_paths = list_images(cfg.DATA.root)
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No images found under {cfg.DATA.root}")

    n_total = len(all_paths)
    n_train = int(0.9 * n_total)
    n_val   = n_total - n_train

    random.shuffle(all_paths)
    train_paths, val_paths = all_paths[:n_train], all_paths[n_train:]

    train_ds = FolderImages(train_paths, cfg.DATA.crop_size, train=True)
    val_ds   = FolderImages(val_paths,   cfg.DATA.crop_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.DATA.batch_size, shuffle=True,
                              num_workers=cfg.DATA.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.DATA.batch_size, shuffle=False,
                              num_workers=cfg.DATA.num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader

