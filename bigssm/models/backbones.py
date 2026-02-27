from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class BackboneOut:
    feats: List[torch.Tensor]
    strides: List[int]
    channels: List[int]


class Backbone(nn.Module):
    def forward(self, x) -> BackboneOut:
        raise NotImplementedError


class TorchvisionBackbone(Backbone):
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        import torchvision.models as M

        if not hasattr(M, name):
            raise ValueError(f"torchvision has no backbone {name}")
        m = getattr(M, name)(weights="DEFAULT" if pretrained else None)
        self.name = name
        self.m = m

        if "resnet" in name:
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1 = m.layer1
            self.layer2 = m.layer2
            self.layer3 = m.layer3
            self.layer4 = m.layer4
            self.strides = [8, 16, 32]
        elif "efficientnet" in name:
            self.features = m.features
            self.strides = [8, 16, 32]
        elif "shufflenet" in name:
            self.conv1 = m.conv1
            self.maxpool = m.maxpool
            self.stage2 = m.stage2
            self.stage3 = m.stage3
            self.stage4 = m.stage4
            self.strides = [8, 16, 32]
        else:
            raise ValueError(f"Backbone {name} adapter not implemented")

    def forward(self, x) -> BackboneOut:
        if "resnet" in self.name:
            x = self.stem(x)
            x = self.layer1(x)
            p3 = self.layer2(x)
            p4 = self.layer3(p3)
            p5 = self.layer4(p4)
            feats = [p3, p4, p5]
        elif "efficientnet" in self.name:
            feats = []
            cur = x
            stride = 1
            for blk in self.features:
                prev = cur
                cur = blk(cur)
                if cur.shape[-1] < prev.shape[-1]:
                    stride *= 2
                if stride in (8, 16, 32):
                    feats.append(cur)
            feats = (feats + [cur] * 3)[-3:]
        elif "shufflenet" in self.name:
            x = self.maxpool(self.conv1(x))
            p3 = self.stage2(x)
            p4 = self.stage3(p3)
            p5 = self.stage4(p4)
            feats = [p3, p4, p5]
        else:
            raise RuntimeError("unreachable")
        ch = [f.shape[1] for f in feats]
        return BackboneOut(feats=feats, strides=self.strides, channels=ch)


class TimmBackbone(Backbone):
    def __init__(self, name: str, pretrained: bool = True, out_indices=(1, 2, 3)):
        super().__init__()
        import timm

        self.m = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.strides = [8, 16, 32]

    def forward(self, x) -> BackboneOut:
        feats = self.m(x)
        ch = [f.shape[1] for f in feats]
        return BackboneOut(feats=feats, strides=self.strides, channels=ch)


class UltralyticsYOLOBackbone(Backbone):
    def __init__(self, weights_path: str):
        super().__init__()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError("ultralytics is required for yolo11 backbone. Install with: pip install ultralytics") from e
        self.yolo = YOLO(weights_path)
        self.model = self.yolo.model
        self.strides = getattr(self.model, "stride", [8, 16, 32])

    def forward(self, x) -> BackboneOut:
        feats = []
        hooks = []
        cache = []

        def hook(_, __, out):
            cache.append(out)

        convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        for m in convs:
            hooks.append(m.register_forward_hook(hook))
        _ = self.model(x)
        for h in hooks:
            h.remove()

        uniq = []
        for f in cache[::-1]:
            s = f.shape[-1]
            if not any(u.shape[-1] == s for u in uniq):
                uniq.append(f)
            if len(uniq) == 3:
                break
        uniq = list(reversed(uniq))
        if len(uniq) != 3:
            last = cache[-1]
            uniq = [last, last, last]
        ch = [f.shape[1] for f in uniq]
        return BackboneOut(feats=uniq, strides=list(self.strides), channels=ch)


def build_backbone(name: str, pretrained: bool = True, yolo_weights: Optional[str] = None) -> Backbone:
    n = name.lower()
    if n in ("yolo11", "yolov11", "yolo-v11"):
        if not yolo_weights:
            raise ValueError("--yolo_weights is required for yolo11 backbone")
        return UltralyticsYOLOBackbone(yolo_weights)

    if n in ("swin_t", "swin_tiny"):
        return TimmBackbone("swin_tiny_patch4_window7_224", pretrained=pretrained)
    if n in ("swin_s", "swin_small"):
        return TimmBackbone("swin_small_patch4_window7_224", pretrained=pretrained)

    # Try timm for other modern backbones
    if n in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "shufflenet_v2_x1_0"):
        try:
            return TimmBackbone(n, pretrained=pretrained)
        except Exception:
            pass

    return TorchvisionBackbone(name=n, pretrained=pretrained)