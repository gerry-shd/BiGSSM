import argparse
import os
import yaml
import torch

from bigssm.engine import evaluate
from bigssm.models import BiGSSMDetector
from bigssm.data import YoloDataset, collate_fn
from bigssm.utils import load_yaml
from torch.utils.data import DataLoader


def resolve_data(data, data_file=None):
    root = data.get("path", "")
    if root and not os.path.isabs(root):
        base = os.path.dirname(os.path.abspath(data_file)) if data_file else os.getcwd()
        root = os.path.abspath(os.path.join(base, root))
    elif not root:
        root = os.path.dirname(os.path.abspath(data_file)) if data_file else os.getcwd()

    def j(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    val_img = j(data["val"])

    def infer_label_dir(img_dir):
        if "images" in img_dir:
            return img_dir.replace("images", "labels")
        return os.path.join(os.path.dirname(img_dir), "labels")

    data["val_img"] = val_img
    data["val_lbl"] = infer_label_dir(val_img)
    return data


def apply_overrides(cfg, opts):
    for opt in opts:
        if "=" not in opt:
            continue
        k, v = opt.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def main(args):
    cfg = apply_overrides(load_yaml(args.cfg), args.opts)
    data = resolve_data(load_yaml(args.data), args.data)

    args.backbone = args.backbone or str(cfg.get("backbone", "resnet50"))
    if args.yolo_weights is None:
        args.yolo_weights = cfg.get("yolo_weights", None)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    nc = int(data["nc"])
    cfg["nc"] = nc

    model = BiGSSMDetector(
        nc=nc,
        backbone=args.backbone,
        pretrained=False,
        yolo_weights=args.yolo_weights,
        width=float(cfg.get("width", 1.0)),
        use_mamba=bool(cfg.get("use_mamba", True)) and (not args.no_mamba),
        reg_max=int(cfg.get("reg_max", 16)),
    ).to(device)

    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("ema", None) or ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    val_ds = YoloDataset(data["val_img"], data["val_lbl"], img_size=int(cfg["img"]), augment=False)
    if len(val_ds) == 0:
        raise ValueError(
            f"Empty val dataset: found 0 images under '{data['val_img']}'. "
            "Check data.yaml path/val fields and image extensions."
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    metrics = evaluate(model, val_loader, cfg, device)
    print(metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--no_mamba", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    main(args)
