from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch

from .utils import bbox_iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iouv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Sort by confidence
    idx = np.argsort(-conf)
    tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]

    unique_classes = np.unique(target_cls)
    ap = np.zeros((unique_classes.size, iouv.size))
    precision = np.zeros((unique_classes.size, iouv.size))
    recall = np.zeros((unique_classes.size, iouv.size))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_gt == 0:
            continue

        tpc = tp[i].astype(np.float32)
        fpc = 1 - tpc
        tpc = np.cumsum(tpc, axis=0)
        fpc = np.cumsum(fpc, axis=0)

        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc + 1e-16)

        for j in range(iouv.size):
            rc = recall_curve[:, j]
            pc = precision_curve[:, j]
            ap[ci, j] = compute_ap(rc, pc)
            f1c = 2 * pc * rc / (pc + rc + 1e-16)
            k = int(np.argmax(f1c))
            precision[ci, j] = pc[k]
            recall[ci, j] = rc[k]

    return precision, recall, ap, unique_classes.astype(int)


def process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor):
    # detections: [N,6] xyxy,conf,cls
    # labels: [M,5] cls,xyxy
    correct = torch.zeros(detections.shape[0], iouv.numel(), dtype=torch.bool, device=detections.device)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou = bbox_iou(labels[:, 1:], detections[:, :4])  # [M,N]
    same_cls = labels[:, 0:1].long() == detections[:, 5][None, :].long()

    for j, t in enumerate(iouv):
        x = torch.where((iou >= t) & same_cls)
        if x[0].numel() == 0:
            continue

        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [K,3]: li,di,iou
        if matches.shape[0] > 1:
            m = matches.cpu().numpy()
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            matches = torch.from_numpy(m).to(detections.device)

        correct[matches[:, 1].long(), j] = True
    return correct
