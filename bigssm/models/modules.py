from __future__ import annotations
import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class SlimStem(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_mid = max(c2 // 4, 16)
        self.cv1 = Conv(c1, c_mid, 3, 2)
        self.cv2 = Conv(c_mid, c_mid, 3, 2)
        self.cv3 = Conv(c_mid, c2, 1, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.cv1(x)))


class PatchUnfoldMergeBlock(nn.Module):
    def __init__(self, c: int, k: int = 3, expand: int = 2):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=k, padding=k // 2, stride=1)
        self.proj = nn.Conv2d(c * (k * k), c * expand, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c * expand)
        self.act = nn.SiLU(inplace=True)
        self.dw = nn.Conv2d(c * expand, c * expand, 3, 1, 1, groups=c * expand, bias=False)
        self.pw = nn.Conv2d(c * expand, c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.unfold.kernel_size
        if isinstance(k, tuple):
            k_h, k_w = k
        else:
            k_h = k_w = int(k)
        p = self.unfold(x).view(b, c * (k_h * k_w), h, w)
        y = self.act(self.bn(self.proj(p)))
        y = self.dw(y)
        y = self.bn2(self.pw(y))
        return x + y


class DWConvResBlock(nn.Module):
    def __init__(self, c: int, expand: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = c * expand
        self.fc1 = nn.Conv2d(c, hidden, 1, 1, 0)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, c, 1, 1, 0)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.fc2(self.drop(self.act(self.bn(self.dw(self.fc1(x))))))
        return x + y


class DWConvGatedBlock(nn.Module):
    def __init__(self, c: int, expand: int = 4):
        super().__init__()
        hidden = c * expand
        self.fc = nn.Conv2d(c, hidden * 2, 1, 1, 0)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(hidden, c, 1, 1, 0)

    def forward(self, x):
        u, v = self.fc(x).chunk(2, dim=1)
        u = self.dw(u)
        u = self.act(u)
        g = torch.sigmoid(v)
        return x + self.proj(u * g)


class _ConvSSM1D(nn.Module):
    def __init__(self, d_model: int, kernel: int = 9):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel - 1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.dw(x)[..., : x.shape[-1]]
        y = self.pw(self.act(y))
        return y.transpose(1, 2)


class BiSS2DLocalGatedBlock(nn.Module):
    def __init__(self, c: int, ssm_kernel: int = 9, use_mamba: bool = True):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(c)
        self.use_mamba = False
        self.mamba = None
        self.ssm = None
        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore

                self.mamba = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)
                self.use_mamba = True
            except Exception:
                self.use_mamba = False
        if not self.use_mamba:
            self.ssm = _ConvSSM1D(d_model=c, kernel=ssm_kernel)

        self.local = DWConv(c, c, 3, 1)
        self.gate = nn.Conv2d(c, c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def _ssm(self, x_seq):
        if self.use_mamba and self.mamba is not None:
            return self.mamba(x_seq)
        return self.ssm(x_seq)  # type: ignore

    def forward(self, x):
        b, c, h, w = x.shape
        x_ln = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        row = x_ln.permute(0, 2, 3, 1).reshape(b * h, w, c)
        row_f = self._ssm(row)
        row_b = torch.flip(self._ssm(torch.flip(row, dims=[1])), dims=[1])
        row = (row_f + row_b).reshape(b, h, w, c).permute(0, 3, 1, 2)

        col = x_ln.permute(0, 3, 2, 1).reshape(b * w, h, c)
        col_f = self._ssm(col)
        col_b = torch.flip(self._ssm(torch.flip(col, dims=[1])), dims=[1])
        col = (col_f + col_b).reshape(b, w, h, c).permute(0, 3, 2, 1)

        ssm_feat = row + col
        local_feat = self.local(x_ln)
        g = torch.sigmoid(self.gate(x_ln))
        y = g * ssm_feat + (1.0 - g) * local_feat
        y = self.proj(y)
        return x + y


class DualBranchFusionBlock(nn.Module):
    def __init__(self, c: int, n: int = 2, use_mamba: bool = True):
        super().__init__()
        blocks = []
        for _ in range(n):
            blocks += [
                BiSS2DLocalGatedBlock(c, use_mamba=use_mamba),
                DWConvResBlock(c, expand=2),
                DWConvGatedBlock(c, expand=2),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.out(self.blocks(x))
