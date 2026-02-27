from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from .backbones import build_backbone
from .head import DetectHead
from .neck import BiGSSMFPN
from ..assigner import make_anchors
from ..utils import non_max_suppression


class BiGSSMDetector(nn.Module):
    def __init__(
        self,
        nc: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        yolo_weights: str | None = None,
        width: float = 1.0,
        use_mamba: bool = True,
        reg_max: int = 16,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.backbone = build_backbone(backbone, pretrained=pretrained, yolo_weights=yolo_weights)
        dummy = self.backbone(torch.zeros(1, 3, 64, 64))
        self.neck = BiGSSMFPN(dummy.channels, width=width, use_mamba=use_mamba)
        ch = [int(x * width) for x in (256, 512, 1024)]
        self.head = DetectHead(ch, nc=nc, reg_max=reg_max)

    def forward(self, x):
        feats = self.backbone(x).feats
        feats = self.neck(feats)
        return self.head(feats)

    @torch.no_grad()
    def predict(self, x, conf_thres=0.25, iou_thres=0.45, max_det=300):
        self.eval()
        outputs = self.forward(x)
        strides = [int(x.shape[-1] / f.shape[-1]) for f in outputs]
        anchor_points, stride_tensor = make_anchors(outputs, strides, x.device)
        preds = []
        reg_max = self.reg_max
        proj = torch.linspace(0, reg_max, reg_max + 1, device=x.device)
        for b in range(x.shape[0]):
            pred = []
            for o in outputs:
                o = o[b].permute(1, 2, 0).reshape(-1, o.shape[1])
                pred.append(o)
            pred = torch.cat(pred, 0)
            pred_dist = pred[:, : 4 * (reg_max + 1)].reshape(-1, 4, reg_max + 1)
            pred_obj = pred[:, 4 * (reg_max + 1) : 4 * (reg_max + 1) + 1]
            pred_cls = pred[:, 4 * (reg_max + 1) + 1 :]
            prob = pred_dist.softmax(dim=-1)
            dist = (prob * proj).sum(dim=-1) * stride_tensor
            boxes = torch.stack(
                [
                    anchor_points[:, 0] - dist[:, 0],
                    anchor_points[:, 1] - dist[:, 1],
                    anchor_points[:, 0] + dist[:, 2],
                    anchor_points[:, 1] + dist[:, 3],
                ],
                dim=-1,
            )
            scores = pred_obj.sigmoid() * pred_cls.sigmoid()
            conf, cls = scores.max(dim=1)
            det = torch.cat([boxes, conf[:, None], cls.float()[:, None]], dim=1)
            det = non_max_suppression(det, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
            preds.append(det)
        return preds