import glob
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import (
    copy_paste,
    flip_lr,
    load_image,
    mixup,
    mosaic,
    random_hsv,
    resize_and_pad,
)


@dataclass
class Sample:
    img: str
    label: str


def load_labels(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                rows.append([float(x) for x in p[:5]])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


class YoloDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        augment: bool = False,
        hsv=(0.015, 0.7, 0.4),
        fliplr: float = 0.5,
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0,
        copy_paste_prob: float = 0.0,
    ):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
        imgs = []
        for e in exts:
            imgs += glob.glob(os.path.join(img_dir, "**", e), recursive=True)
        imgs = sorted(imgs)
        self.samples: List[Sample] = []
        for p in imgs:
            name = os.path.splitext(os.path.basename(p))[0]
            lp = os.path.join(label_dir, name + ".txt")
            self.samples.append(Sample(p, lp))
        self.img_size = img_size
        self.augment = augment
        self.hsv = hsv
        self.fliplr = fliplr
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.augment and random.random() < self.mosaic_prob:
            indices = [idx] + random.sample(range(len(self.samples)), 3)
            imgs = []
            labels = []
            for i in indices:
                s = self.samples[i]
                img = load_image(s.img)
                lb = load_labels(s.label)
                imgs.append(img)
                labels.append(lb)
            img, lb = mosaic(imgs, labels, self.img_size)
            if self.mixup_prob > 0 and random.random() < self.mixup_prob:
                s2 = self.samples[random.randint(0, len(self.samples) - 1)]
                img2 = load_image(s2.img)
                lb2 = load_labels(s2.label)
                img2, lb2 = resize_and_pad(img2, lb2, self.img_size)
                img, lb = mixup(img, lb, img2, lb2)
        else:
            s = self.samples[idx]
            img = load_image(s.img)
            lb = load_labels(s.label)
            img, lb = resize_and_pad(img, lb, self.img_size)

        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            s2 = self.samples[random.randint(0, len(self.samples) - 1)]
            img2 = load_image(s2.img)
            lb2 = load_labels(s2.label)
            img2, lb2 = resize_and_pad(img2, lb2, self.img_size)
            img, lb = copy_paste(img, lb, img2, lb2, p=1.0)

        if self.augment:
            img = random_hsv(img, *self.hsv)
            if random.random() < self.fliplr:
                img, lb = flip_lr(img, lb)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        if lb.size:
            labels_px = lb.copy()
            labels_px[:, 1] *= self.img_size
            labels_px[:, 2] *= self.img_size
            labels_px[:, 3] *= self.img_size
            labels_px[:, 4] *= self.img_size
        else:
            labels_px = np.zeros((0, 5), dtype=np.float32)

        return torch.from_numpy(img), torch.from_numpy(labels_px), s.img


def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = []
    for i, lb in enumerate(labels):
        if lb.numel() == 0:
            continue
        img_idx = torch.full((lb.shape[0], 1), i, dtype=lb.dtype)
        targets.append(torch.cat([img_idx, lb], dim=1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6), dtype=torch.float32)
    return imgs, targets, list(paths)
