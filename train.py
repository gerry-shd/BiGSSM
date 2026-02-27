import argparse
import os
import yaml

from bigssm.engine import train
from bigssm.utils import load_yaml


def resolve_data(data, data_file=None):
    root = data.get("path", "")
    if root and not os.path.isabs(root):
        base = os.path.dirname(os.path.abspath(data_file)) if data_file else os.getcwd()
        root = os.path.abspath(os.path.join(base, root))
    elif not root:
        root = os.path.dirname(os.path.abspath(data_file)) if data_file else os.getcwd()

    def j(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    train_img = j(data["train"])
    val_img = j(data["val"])

    def infer_label_dir(img_dir):
        if "images" in img_dir:
            return img_dir.replace("images", "labels")
        return os.path.join(os.path.dirname(img_dir), "labels")

    data["train_img"] = train_img
    data["val_img"] = val_img
    data["train_lbl"] = infer_label_dir(train_img)
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
    cfg = load_yaml(args.cfg)
    cfg = apply_overrides(cfg, args.opts)
    data = resolve_data(load_yaml(args.data), args.data)

    args.backbone = args.backbone or str(cfg.get("backbone", "resnet50"))
    if args.weights is None:
        args.weights = cfg.get("weights", None)
    if args.yolo_weights is None:
        args.yolo_weights = cfg.get("yolo_weights", None)
    if not bool(cfg.get("pretrained", True)):
        args.no_pretrained = True

    train(cfg, data, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/gerry/datas/cones_obj_yolo_v5/dataset.yaml")
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--project", default="runs/exp")
    ap.add_argument("--exist_ok", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_mamba", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    main(args)
