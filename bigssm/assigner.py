from __future__ import annotations
from typing import Tuple
import torch

from .utils import bbox_iou


def make_anchors(feats, strides, device):
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[-2:]
        sy = torch.arange(h, device=device) + 0.5
        sx = torch.arange(w, device=device) + 0.5
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        points = torch.stack((gx, gy), dim=-1).reshape(-1, 2) * stride
        anchor_points.append(points)
        stride_tensor.append(torch.full((points.shape[0], 1), stride, device=device))
    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    x1 = anchor_points[:, 0] - distance[:, 0]
    y1 = anchor_points[:, 1] - distance[:, 1]
    x2 = anchor_points[:, 0] + distance[:, 2]
    y2 = anchor_points[:, 1] + distance[:, 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox2dist(anchor_points: torch.Tensor, bboxes: torch.Tensor, reg_max: int) -> torch.Tensor:
    left = anchor_points[:, 0] - bboxes[:, 0]
    top = anchor_points[:, 1] - bboxes[:, 1]
    right = bboxes[:, 2] - anchor_points[:, 0]
    bottom = bboxes[:, 3] - anchor_points[:, 1]
    dist = torch.stack((left, top, right, bottom), dim=-1)
    return dist.clamp(min=0, max=reg_max - 1e-3)


class TaskAlignedAssigner:
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]
        device = pred_bboxes.device

        target_labels = torch.full((num_pred,), -1, device=device, dtype=torch.long)
        target_bboxes = torch.zeros((num_pred, 4), device=device)
        target_scores = torch.zeros((num_pred, pred_scores.shape[1]), device=device)
        fg_mask = torch.zeros((num_pred,), device=device, dtype=torch.bool)

        if num_gt == 0 or num_pred == 0:
            return target_labels, target_bboxes, target_scores, fg_mask

        ious = bbox_iou(gt_bboxes, pred_bboxes)  # [M,N]
        cls_scores = pred_scores[:, gt_labels]  # [N,M]
        cls_scores = cls_scores.transpose(0, 1)  # [M,N]

        ap = anchor_points
        in_gt = (
            (ap[:, 0][None, :] >= gt_bboxes[:, 0:1])
            & (ap[:, 0][None, :] <= gt_bboxes[:, 2:3])
            & (ap[:, 1][None, :] >= gt_bboxes[:, 1:2])
            & (ap[:, 1][None, :] <= gt_bboxes[:, 3:4])
        )

        align_metric = (cls_scores.clamp(0, 1) ** self.alpha) * (ious.clamp(0, 1) ** self.beta)
        align_metric = align_metric * in_gt.float()

        topk = min(self.topk, num_pred)
        topk_idx = torch.topk(align_metric, k=topk, dim=1, largest=True).indices  # [M,topk]
        pos_mask = torch.zeros_like(align_metric, dtype=torch.bool)
        for gi in range(num_gt):
            pos_mask[gi, topk_idx[gi]] = True
        pos_mask &= in_gt

        ious_pos = ious * pos_mask.float()
        max_iou, gt_idx = ious_pos.max(dim=0)
        fg_mask = max_iou > 0
        if fg_mask.any():
            assigned_gt = gt_idx[fg_mask]
            target_bboxes[fg_mask] = gt_bboxes[assigned_gt]
            target_labels[fg_mask] = gt_labels[assigned_gt]
            quality = max_iou[fg_mask].clamp(0, 1)
            target_scores[fg_mask, target_labels[fg_mask]] = quality

        return target_labels, target_bboxes, target_scores, fg_mask


class SimOTAAssigner:
    def __init__(self, candidate_topk: int = 10, iou_weight: float = 3.0, topk: int = 10):
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.topk = topk

    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_gt = gt_bboxes.shape[0]
        num_pred = pred_bboxes.shape[0]
        device = pred_bboxes.device

        target_labels = torch.full((num_pred,), -1, device=device, dtype=torch.long)
        target_bboxes = torch.zeros((num_pred, 4), device=device)
        target_scores = torch.zeros((num_pred, pred_scores.shape[1]), device=device)
        fg_mask = torch.zeros((num_pred,), device=device, dtype=torch.bool)

        if num_gt == 0 or num_pred == 0:
            return target_labels, target_bboxes, target_scores, fg_mask

        ious = bbox_iou(gt_bboxes, pred_bboxes)  # [M,N]
        cls_scores = pred_scores[:, gt_labels].transpose(0, 1).clamp(1e-6, 1 - 1e-6)

        ap = anchor_points
        in_gt = (
            (ap[:, 0][None, :] >= gt_bboxes[:, 0:1])
            & (ap[:, 0][None, :] <= gt_bboxes[:, 2:3])
            & (ap[:, 1][None, :] >= gt_bboxes[:, 1:2])
            & (ap[:, 1][None, :] <= gt_bboxes[:, 3:4])
        )

        cost = -torch.log(cls_scores) + self.iou_weight * (1 - ious)
        cost = cost + (~in_gt) * 1e6

        topk = min(self.candidate_topk, num_pred)
        candidate_idx = torch.topk(ious, k=topk, dim=1, largest=True).indices

        dynamic_ks = []
        for gi in range(num_gt):
            iou_topk = ious[gi, candidate_idx[gi]]
            dynamic_ks.append(max(int(iou_topk.sum().item()), 1))

        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
        for gi in range(num_gt):
            k = dynamic_ks[gi]
            _, idx = torch.topk(cost[gi], k=k, largest=False)
            matching_matrix[gi, idx] = True

        matched_cost = cost + (~matching_matrix) * 1e6
        min_cost, gt_idx = matched_cost.min(dim=0)
        fg_mask = min_cost < 1e6
        if fg_mask.any():
            assigned_gt = gt_idx[fg_mask]
            target_bboxes[fg_mask] = gt_bboxes[assigned_gt]
            target_labels[fg_mask] = gt_labels[assigned_gt]
            fg_idx = torch.where(fg_mask)[0]
            quality = ious[assigned_gt, fg_idx].clamp(0, 1)
            target_scores[fg_mask, target_labels[fg_mask]] = quality

        return target_labels, target_bboxes, target_scores, fg_mask