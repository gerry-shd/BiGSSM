import os
import math
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def increment_path(path: str, exist_ok: bool = False) -> str:
    if exist_ok or not os.path.exists(path):
        return path
    base = path
    i = 1
    while True:
        p = f"{base}{i}"
        if not os.path.exists(p):
            return p
        i += 1


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.ndim == 1:
        box1 = box1[None, :]
    if box2.ndim == 1:
        box2 = box2[None, :]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter = (
        (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(0)
        * (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)
    )
    area1 = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    area2 = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    return inter / (area1[:, None] + area2 - inter + eps)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.ndim == 1:
        box1 = box1[None, :]
    if box2.ndim == 1:
        box2 = box2[None, :]
    aligned = box1.shape[0] == box2.shape[0]

    if aligned:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter = (
            (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
            * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        )
        area1 = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
        area2 = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
        iou = inter / (area1 + area2 - inter + eps)

        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        c2 = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2) + eps

        b1_cx = (b1_x1 + b1_x2) / 2
        b1_cy = (b1_y1 + b1_y2) / 2
        b2_cx = (b2_x1 + b2_x2) / 2
        b2_cy = (b2_y1 + b2_y2) / 2
        rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)

        b1_w = (b1_x2 - b1_x1).clamp(0)
        b1_h = (b1_y2 - b1_y1).clamp(0)
        b2_w = (b2_x2 - b2_x1).clamp(0)
        b2_h = (b2_y2 - b2_y1).clamp(0)
        v = (4 / math.pi**2) * (torch.atan(b2_w / (b2_h + eps)) - torch.atan(b1_w / (b1_h + eps))).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return iou - (rho2 / c2 + alpha * v)

    # matrix mode
    iou = bbox_iou(box1, box2, eps=eps)
    return iou


def clip_boxes(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes[..., 0].clamp_(0, w)
    boxes[..., 2].clamp_(0, w)
    boxes[..., 1].clamp_(0, h)
    boxes[..., 3].clamp_(0, h)
    return boxes


def non_max_suppression(
    preds: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
):
    if preds.numel() == 0:
        return preds
    scores = preds[:, 4]
    keep = scores > conf_thres
    preds = preds[keep]
    if preds.numel() == 0:
        return preds
    boxes = preds[:, :4]
    scores = preds[:, 4]
    cls = preds[:, 5]

    keep_idx = []
    for c in cls.unique():
        idx = (cls == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        b = boxes[idx]
        s = scores[idx]
        order = s.argsort(descending=True)
        while order.numel() > 0:
            i = order[0]
            keep_idx.append(idx[i].item())
            if order.numel() == 1:
                break
            iou = bbox_iou(b[i], b[order[1:]]).squeeze(0)
            order = order[1:][iou <= iou_thres]
    keep_idx = torch.tensor(keep_idx, device=preds.device, dtype=torch.long)
    preds = preds[keep_idx]
    if preds.shape[0] > max_det:
        preds = preds[preds[:, 4].argsort(descending=True)[:max_det]]
    return preds


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flops(model: torch.nn.Module, img: int) -> float:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return 0.0
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img, img, device=device)
    flops, _ = profile(model, inputs=(x,), verbose=False)
    return float(flops)