import argparse
import torch

from bigssm.models import BiGSSMDetector
from bigssm.utils import load_yaml


def main(args):
    cfg = load_yaml(args.cfg)
    data = load_yaml(args.data)
    args.backbone = args.backbone or str(cfg.get("backbone", "resnet50"))
    if args.yolo_weights is None:
        args.yolo_weights = cfg.get("yolo_weights", None)

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
    model.eval()

    img = int(cfg.get("img", 640))
    x = torch.zeros(1, 3, img, img, device=device)

    torch.onnx.export(
        model,
        x,
        args.onnx,
        input_names=["images"],
        output_names=["p3", "p4", "p5"],
        dynamic_axes={"images": {0: "batch"}, "p3": {0: "batch"}, "p4": {0: "batch"}, "p5": {0: "batch"}},
        opset_version=12,
    )

    if args.torchscript:
        ts = torch.jit.trace(model, x)
        ts.save(args.torchscript)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--torchscript", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--no_mamba", action="store_true")
    args = ap.parse_args()
    main(args)
