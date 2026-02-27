from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import bbox2dist
from .utils import bbox_ciou


class DynFocusLoss(nn.Module):
    def __init__(self, eta: float = 2.0, alpha: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.eta = eta
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, iou: torch.Tensor | None = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal = (1 - p_t).clamp(min=0) ** self.eta
        loss = bce * focal
        if iou is not None:
            w = ((1.0 - iou).clamp(min=0) + self.eps) ** self.alpha
            loss = loss * w[:, None]
        return loss.mean()


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        weight = self.alpha * pred.pow(self.gamma) * (1 - targets) + targets
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weight


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pred = torch.sigmoid(logits)
        p_t = targets * pred + (1 - targets) * (1 - pred)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = bce * alpha_t * (1 - p_t) ** self.gamma
        return loss


class QualityFocalLoss(nn.Module):
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * (targets - pred).abs() ** self.beta
        return loss


def dfl_loss(pred_dist: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    # pred_dist: [N, 4, reg_max+1]
    # target: [N, 4]
    target = target.clamp(min=0, max=reg_max - 1e-3)
    left = target.floor().long()
    right = left + 1
    weight_right = target - left.float()
    weight_left = 1.0 - weight_right

    pred = pred_dist.reshape(-1, reg_max + 1)
    left = left.reshape(-1)
    right = right.reshape(-1)
    weight_left = weight_left.reshape(-1)
    weight_right = weight_right.reshape(-1)

    loss = F.cross_entropy(pred, left, reduction="none") * weight_left + F.cross_entropy(pred, right, reduction="none") * weight_right
    return loss.reshape(-1, 4).mean(dim=1)


class DetectionLoss(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.nc = int(cfg["nc"])
        self.reg_max = int(cfg["reg_max"])
        self.box_weight = float(cfg.get("box_weight", 7.5))
        self.cls_weight = float(cfg.get("cls_weight", 0.5))
        self.obj_weight = float(cfg.get("obj_weight", 1.0))
        self.dfl_weight = float(cfg.get("dfl_weight", 1.5))
        self.use_dfl = bool(cfg.get("use_dfl", True))
        self.use_ciou = bool(cfg.get("use_ciou", True))

        cls_loss = cfg.get("cls_loss", "dynfocus").lower()
        if cls_loss == "varifocal":
            self.cls_loss = VarifocalLoss(alpha=float(cfg.get("varifocal_alpha", 0.75)), gamma=float(cfg.get("varifocal_gamma", 2.0)))
        elif cls_loss == "focal":
            self.cls_loss = FocalLoss(alpha=float(cfg.get("focal_alpha", 0.25)), gamma=float(cfg.get("focal_gamma", 2.0)))
        else:
            self.cls_loss = DynFocusLoss(eta=float(cfg.get("dynfocus_eta", 2.0)), alpha=float(cfg.get("dynfocus_alpha", 1.0)))

        obj_loss = cfg.get("obj_loss", "bce").lower()
        if obj_loss == "qfl":
            self.obj_loss = QualityFocalLoss(beta=float(cfg.get("qfl_beta", 2.0)))
        else:
            self.obj_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_obj: torch.Tensor,
        pred_cls: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = pred_dist.device
        if fg_mask.any():
            pred_dist_pos = pred_dist[fg_mask]
            anchor_pos = anchor_points[fg_mask]
            stride_pos = stride_tensor[fg_mask]
            target_pos = target_bboxes[fg_mask]

            pred_prob = pred_dist_pos.softmax(dim=-1)
            proj = torch.linspace(0, self.reg_max, self.reg_max + 1, device=device)
            dist = (pred_prob * proj).sum(dim=-1) * stride_pos
            pred_box = torch.stack(
                [
                    anchor_pos[:, 0] - dist[:, 0],
                    anchor_pos[:, 1] - dist[:, 1],
                    anchor_pos[:, 0] + dist[:, 2],
                    anchor_pos[:, 1] + dist[:, 3],
                ],
                dim=-1,
            )
            if self.use_ciou:
                iou = bbox_ciou(pred_box, target_pos).clamp(-1, 1)
                iou_loss = (1.0 - iou).mean()
            else:
                iou = bbox_ciou(pred_box, target_pos).clamp(-1, 1)
                iou_loss = (1.0 - iou).mean()

            if self.use_dfl:
                # DFL targets must be in stride-normalized units before clamping to reg_max bins.
                target_dist = bbox2dist(anchor_pos / stride_pos, target_pos / stride_pos, self.reg_max)
                dfl = dfl_loss(pred_dist_pos, target_dist, self.reg_max).mean()
            else:
                dfl = torch.tensor(0.0, device=device)
        else:
            iou_loss = torch.tensor(0.0, device=device)
            dfl = torch.tensor(0.0, device=device)

        # cls loss
        cls_loss = self.cls_loss(pred_cls, target_scores)
        if cls_loss.ndim > 0:
            cls_loss = cls_loss.mean()

        # obj loss
        obj_target = target_scores.max(dim=1, keepdim=True).values
        obj_loss = self.obj_loss(pred_obj, obj_target)
        if obj_loss.ndim > 0:
            obj_loss = obj_loss.mean()

        total = self.box_weight * iou_loss + self.dfl_weight * dfl + self.cls_weight * cls_loss + self.obj_weight * obj_loss
        return total, {"box": iou_loss.detach(), "dfl": dfl.detach(), "cls": cls_loss.detach(), "obj": obj_loss.detach(), "total": total.detach()}
