import argparse
import json
import os
from pathlib import Path

import cv2
import torch

from bigssm.models import BiGSSMDetector
from bigssm.data.augment import letterbox
from bigssm.utils import load_yaml


def preprocess(img, img_size):
    img0 = img.copy()
    img, r, (dw, dh) = letterbox(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return img, (r, dw, dh), img0


def scale_coords(boxes, r, dw, dh, shape):
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= r
    boxes[:, 0].clamp_(0, shape[1])
    boxes[:, 2].clamp_(0, shape[1])
    boxes[:, 1].clamp_(0, shape[0])
    boxes[:, 3].clamp_(0, shape[0])
    return boxes


def draw_boxes(img, det, names):
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[int(cls)]} {conf:.2f}" if names else f"{int(cls)} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main(args):
    cfg = load_yaml(args.cfg)
    data = load_yaml(args.data)
    args.backbone = args.backbone or str(cfg.get("backbone", "resnet50"))
    if args.yolo_weights is None:
        args.yolo_weights = cfg.get("yolo_weights", None)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = BiGSSMDetector(
        nc=int(data["nc"]),
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

    p = Path(args.source)
    files = [p] if p.is_file() else list(p.glob("*"))
    os.makedirs(args.save_dir, exist_ok=True)
    results = {}

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        x, (r, dw, dh), img0 = preprocess(img, int(cfg["img"]))
        x = x.to(device)
        preds = model.predict(x, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det)[0]
        if preds.numel():
            preds[:, :4] = scale_coords(preds[:, :4], r, dw, dh, img0.shape)
        out_img = draw_boxes(img0.copy(), preds, names)
        out_path = os.path.join(args.save_dir, fp.name)
        cv2.imwrite(out_path, out_img)

        items = []
        for *xyxy, conf, cls in preds.tolist():
            items.append({"bbox": xyxy, "score": conf, "label": int(cls)})
        results[fp.name] = items

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--save_dir", default="runs/predict")
    ap.add_argument("--save_json", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--no_mamba", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max_det", type=int, default=300)
    args = ap.parse_args()
    main(args)
