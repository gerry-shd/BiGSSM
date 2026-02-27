from __future__ import annotations
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import ap_per_class, process_batch
from ..utils import count_params, get_flops, xywh2xyxy


def _measure_fps(model, imgs, cfg, device, repeat=5):
    if imgs is None:
        return 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        _ = model.predict(imgs, conf_thres=cfg.get("conf_thres", 0.001), iou_thres=cfg.get("iou_thres", 0.65))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time() - t0
    return (imgs.shape[0] * repeat) / (t + 1e-9)


@torch.no_grad()
def evaluate(model, dataloader, cfg: Dict, device: torch.device) -> Dict:
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []

    t_inf = 0.0
    n_samples = 0
    first_batch = None

    pbar = tqdm(dataloader, desc="val", leave=False)
    for imgs, targets, _ in pbar:
        imgs = imgs.to(device)
        targets = targets.to(device)
        if first_batch is None:
            first_batch = imgs

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        preds = model.predict(imgs, conf_thres=cfg.get("conf_thres", 0.001), iou_thres=cfg.get("iou_thres", 0.65))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        n_samples += imgs.shape[0]
        pbar.set_postfix({"samples": n_samples, "fps": f"{n_samples / (t_inf + 1e-9):.1f}"})

        for si, det in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            if labels.numel():
                labels_xyxy = xywh2xyxy(labels[:, 1:5])
                labels = torch.cat([labels[:, 0:1], labels_xyxy], dim=1)
            correct = process_batch(det, labels, iouv)
            stats.append((correct.cpu(), det[:, 4].cpu(), det[:, 5].cpu(), labels[:, 0].cpu()))

    if len(stats):
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        precision, recall, ap, ap_class = ap_per_class(*stats, iouv=iouv.cpu().numpy())
        ap50 = ap[:, 0]
        map50 = ap50.mean() if ap50.size else 0.0
        map = ap.mean() if ap.size else 0.0
        p = precision[:, 0].mean() if precision.size else 0.0
        r = recall[:, 0].mean() if recall.size else 0.0
        f1 = 2 * p * r / (p + r + 1e-16)
    else:
        map50 = map = p = r = f1 = 0.0
        ap = np.zeros((0, iouv.numel()))
        ap_class = np.array([])

    fps_b = n_samples / (t_inf + 1e-9)
    fps_b1 = 0.0
    params = count_params(model)
    flops = 0.0
    if cfg.get("profile_eval", False):
        if first_batch is not None:
            fps_b1 = _measure_fps(model, first_batch[:1], cfg, device, repeat=5)
        flops = get_flops(model, int(cfg.get("img", 640)))

    per_class_ap = {int(c): float(ap[i].mean()) for i, c in enumerate(ap_class)}

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "map_50": float(map50),
        "map_50_95": float(map),
        "per_class_ap": per_class_ap,
        "fps_b1": float(fps_b1),
        "fps": float(fps_b),
        "params": int(params),
        "flops": float(flops),
    }
