from __future__ import annotations
import torch
import torch.nn as nn

from .modules import Conv, DWConv, DualBranchFusionBlock, PatchUnfoldMergeBlock


class BiGSSMFPN(nn.Module):
    def __init__(self, ch, width=1.0, use_mamba=True):
        super().__init__()
        c3, c4, c5 = ch
        c3n, c4n, c5n = [int(x * width) for x in (256, 512, 1024)]

        self.lateral3 = Conv(c3, c3n, 1, 1)
        self.lateral4 = Conv(c4, c4n, 1, 1)
        self.lateral5 = Conv(c5, c5n, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse4 = Conv(c4n + c5n, c4n, 3, 1)
        self.fuse3 = Conv(c3n + c4n, c3n, 3, 1)

        self.db4 = DualBranchFusionBlock(c4n, n=1, use_mamba=use_mamba)
        self.db3 = DualBranchFusionBlock(c3n, n=1, use_mamba=use_mamba)
        self.db5 = DualBranchFusionBlock(c5n, n=1, use_mamba=use_mamba)

        self.down3 = DWConv(c3n, c4n, 3, 2)
        self.down4 = DWConv(c4n, c5n, 3, 2)
        self.pan4 = Conv(c4n + c4n, c4n, 3, 1)
        self.pan5 = Conv(c5n + c5n, c5n, 3, 1)

        self.pum3 = PatchUnfoldMergeBlock(c3n, k=3, expand=2)
        self.pum4 = PatchUnfoldMergeBlock(c4n, k=3, expand=2)
        self.pum5 = PatchUnfoldMergeBlock(c5n, k=3, expand=2)

    def forward(self, feats):
        c3, c4, c5 = feats
        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)
        p5 = self.lateral5(c5)

        p4 = self.fuse4(torch.cat([p4, self.upsample(p5)], dim=1))
        p4 = self.db4(p4)
        p3 = self.fuse3(torch.cat([p3, self.upsample(p4)], dim=1))
        p3 = self.db3(p3)

        n4 = self.pan4(torch.cat([p4, self.down3(p3)], dim=1))
        n5 = self.pan5(torch.cat([p5, self.down4(n4)], dim=1))

        p3 = self.pum3(p3)
        n4 = self.pum4(n4)
        n5 = self.pum5(n5)

        return [p3, n4, n5]