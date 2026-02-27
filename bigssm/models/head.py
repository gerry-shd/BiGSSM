from __future__ import annotations
import torch
import torch.nn as nn

from .modules import Conv


class DetectHead(nn.Module):
    def __init__(self, ch, nc: int, reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for c in ch:
            self.stems.append(Conv(c, c, 3, 1))
            self.cls_convs.append(nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1)))
            self.reg_convs.append(nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1)))
            self.cls_preds.append(nn.Conv2d(c, nc, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(c, 4 * (reg_max + 1), 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(c, 1, 1, 1, 0))

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)
            cls_out = self.cls_preds[i](cls_feat)
            reg_out = self.reg_preds[i](reg_feat)
            obj_out = self.obj_preds[i](reg_feat)
            out = torch.cat([reg_out, obj_out, cls_out], dim=1)
            outs.append(out)
        return outs