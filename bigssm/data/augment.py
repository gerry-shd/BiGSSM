import random
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def random_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def flip_lr(img, labels):
    img = np.ascontiguousarray(np.fliplr(img))
    if labels.size:
        labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels


def resize_and_pad(img, labels, img_size):
    h0, w0 = img.shape[:2]
    img, r, (dw, dh) = letterbox(img, (img_size, img_size))
    if labels.size:
        labels = labels.copy()
        labels[:, 1] *= w0
        labels[:, 3] *= w0
        labels[:, 2] *= h0
        labels[:, 4] *= h0
        labels[:, 1] = labels[:, 1] * r + dw
        labels[:, 2] = labels[:, 2] * r + dh
        labels[:, 3] = labels[:, 3] * r
        labels[:, 4] = labels[:, 4] * r
        labels[:, 1] /= img_size
        labels[:, 2] /= img_size
        labels[:, 3] /= img_size
        labels[:, 4] /= img_size
    return img, labels


def mosaic(imgs, labels, img_size):
    mosaic_img = np.full((img_size * 2, img_size * 2, 3), 114, dtype=np.uint8)
    yc = int(random.uniform(img_size * 0.5, img_size * 1.5))
    xc = int(random.uniform(img_size * 0.5, img_size * 1.5))

    mosaic_labels = []
    for i in range(4):
        img = imgs[i]
        h, w = img.shape[:2]
        labels_i = labels[i].copy() if labels[i].size else labels[i]

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        if labels_i.size:
            labels_i[:, 1] *= w
            labels_i[:, 3] *= w
            labels_i[:, 2] *= h
            labels_i[:, 4] *= h
            labels_i[:, 1] = labels_i[:, 1] + padw
            labels_i[:, 2] = labels_i[:, 2] + padh
            mosaic_labels.append(labels_i)

    if mosaic_labels:
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)
    else:
        mosaic_labels = np.zeros((0, 5), dtype=np.float32)

    x1 = max(xc - img_size // 2, 0)
    y1 = max(yc - img_size // 2, 0)
    x2 = x1 + img_size
    y2 = y1 + img_size
    mosaic_img = mosaic_img[y1:y2, x1:x2]

    if mosaic_labels.size:
        mosaic_labels[:, 1] -= x1
        mosaic_labels[:, 2] -= y1
        mosaic_labels[:, 1] = mosaic_labels[:, 1].clip(0, img_size)
        mosaic_labels[:, 2] = mosaic_labels[:, 2].clip(0, img_size)
        mosaic_labels[:, 3] = mosaic_labels[:, 3].clip(0, img_size)
        mosaic_labels[:, 4] = mosaic_labels[:, 4].clip(0, img_size)
        # normalize
        mosaic_labels[:, 1] /= img_size
        mosaic_labels[:, 2] /= img_size
        mosaic_labels[:, 3] /= img_size
        mosaic_labels[:, 4] /= img_size

    return mosaic_img, mosaic_labels


def mixup(img1, labels1, img2, labels2):
    r = np.random.beta(32.0, 32.0)
    img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    if labels1.size and labels2.size:
        labels = np.concatenate([labels1, labels2], axis=0)
    elif labels1.size:
        labels = labels1
    else:
        labels = labels2
    return img, labels


def copy_paste(img, labels, img2, labels2, p=0.5):
    if labels2.size == 0 or random.random() > p:
        return img, labels
    h, w = img.shape[:2]
    labels_new = [labels]
    for lb in labels2:
        cls, cx, cy, bw, bh = lb
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue
        img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        labels_new.append(
            np.array([[cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]], dtype=np.float32)
        )
    if labels_new:
        labels = np.concatenate(labels_new, axis=0)
    return img, labels