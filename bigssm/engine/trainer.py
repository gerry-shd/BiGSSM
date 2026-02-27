from __future__ import annotations
import math
import os
import random
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..assigner import SimOTAAssigner, TaskAlignedAssigner, make_anchors
from ..data import YoloDataset, collate_fn
from ..losses import DetectionLoss
from ..utils import increment_path, set_seed, xywh2xyxy
from .ema import ModelEMA
from .evaluator import evaluate


def build_optimizer(model, cfg):
    lr0 = float(cfg["lr0"])
    momentum = float(cfg["momentum"])
    wd = float(cfg["weight_decay"])
    g0, g1 = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            g0.append(p)
        else:
            g1.append(p)
    opt = torch.optim.SGD(g0, lr=lr0, momentum=momentum, nesterov=True)
    opt.add_param_group({"params": g1, "weight_decay": wd})
    return opt


def decode_bboxes(pred_dist, anchor_points, stride_tensor, reg_max):
    proj = torch.linspace(0, reg_max, reg_max + 1, device=pred_dist.device)
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
    return boxes


def resize_batch(imgs, targets, new_size, old_size):
    if new_size == old_size:
        return imgs, targets
    scale = new_size / old_size
    imgs = F.interpolate(imgs, size=(new_size, new_size), mode="bilinear", align_corners=False)
    targets = targets.clone()
    targets[:, 2:6] *= scale
    return imgs, targets


def train(cfg: Dict, data: Dict, args):
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    img_size = int(cfg["img"])
    nc = int(data["nc"])
    cfg["nc"] = nc

    train_ds = YoloDataset(
        data["train_img"],
        data["train_lbl"],
        img_size=img_size,
        augment=True,
        hsv=(cfg["hsv_h"], cfg["hsv_s"], cfg["hsv_v"]),
        fliplr=cfg["fliplr"],
        mosaic_prob=cfg.get("mosaic", 0.0),
        mixup_prob=cfg.get("mixup", 0.0),
        copy_paste_prob=cfg.get("copy_paste", 0.0),
    )
    val_ds = YoloDataset(
        data["val_img"],
        data["val_lbl"],
        img_size=img_size,
        augment=False,
    )

    if len(train_ds) == 0:
        raise ValueError(
            f"Empty train dataset: found 0 images under '{data['train_img']}'. "
            f"Check data.yaml path/train fields and image extensions."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"Empty val dataset: found 0 images under '{data['val_img']}'. "
            f"Check data.yaml path/val fields and image extensions."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch"]),
        shuffle=True,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch"]),
        shuffle=False,
        num_workers=int(cfg["workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    from ..models import BiGSSMDetector

    model = BiGSSMDetector(
        nc=nc,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        yolo_weights=args.yolo_weights,
        width=float(cfg.get("width", 1.0)),
        use_mamba=bool(cfg.get("use_mamba", True)) and (not args.no_mamba),
        reg_max=int(cfg.get("reg_max", 16)),
    ).to(device)

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    elif args.yolo_weights:
        # attempt to warm-start with YOLOv11 weights if compatible
        try:
            ckpt = torch.load(args.yolo_weights, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=False)
            print("[info] loaded yolo11 weights with strict=False")
        except Exception as e:
            print(f"[warn] failed to load yolo11 weights: {e}")

    opt = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.get("use_amp", True)))
    ema = ModelEMA(model, decay=float(cfg.get("ema_decay", 0.9999))) if cfg.get("use_ema", True) else None

    epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 3))

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        t = (e - warmup) / max(epochs - warmup, 1)
        return cfg["lrf"] + 0.5 * (1 - cfg["lrf"]) * (1 + math.cos(math.pi * t))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    assigner_name = str(cfg.get("assigner", "taskaligned")).lower()
    if assigner_name == "simota":
        assigner = SimOTAAssigner(
            candidate_topk=int(cfg.get("simota_candidate_topk", 10)),
            iou_weight=float(cfg.get("simota_iou_weight", 3.0)),
            topk=int(cfg.get("simota_topk", 10)),
        )
    else:
        assigner = TaskAlignedAssigner(
            topk=int(cfg.get("taskaligned_topk", 10)),
            alpha=float(cfg.get("taskaligned_alpha", 0.5)),
            beta=float(cfg.get("taskaligned_beta", 6.0)),
        )

    criterion = DetectionLoss(cfg).to(device)

    project = increment_path(args.project, exist_ok=args.exist_ok)
    os.makedirs(os.path.join(project, "weights"), exist_ok=True)
    log_path = os.path.join(project, "train.log")

    best = -1.0
    val_interval = max(int(cfg.get("val_interval", 1)), 1)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}")
        opt.zero_grad(set_to_none=True)

        for ni, (imgs, targets, _) in enumerate(pbar):
            imgs = imgs.to(device)
            targets = targets.to(device)

            if cfg.get("multi_scale", True):
                ms_min, ms_max = cfg.get("ms_range", [0.5, 1.5])
                new_size = int(img_size * random.uniform(ms_min, ms_max) // 32 * 32)
                imgs, targets = resize_batch(imgs, targets, new_size, img_size)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(imgs)
                strides = [int(imgs.shape[-1] / o.shape[-2]) for o in outputs]
                anchor_points, stride_tensor = make_anchors(outputs, strides, imgs.device)

                loss_total = 0.0
                logs = {"box": 0.0, "dfl": 0.0, "cls": 0.0, "obj": 0.0, "total": 0.0}

                for bi in range(imgs.shape[0]):
                    pred = []
                    for o in outputs:
                        o = o[bi].permute(1, 2, 0).reshape(-1, o.shape[1])
                        pred.append(o)
                    pred = torch.cat(pred, 0)
                    reg_max = int(cfg.get("reg_max", 16))
                    pred_dist = pred[:, : 4 * (reg_max + 1)].reshape(-1, 4, reg_max + 1)
                    pred_obj = pred[:, 4 * (reg_max + 1) : 4 * (reg_max + 1) + 1]
                    pred_cls = pred[:, 4 * (reg_max + 1) + 1 :]

                    pred_cls_prob = pred_cls.sigmoid()
                    if bool(cfg.get("assigner_use_obj", True)):
                        pred_scores = pred_cls_prob * pred_obj.sigmoid()
                    else:
                        pred_scores = pred_cls_prob
                    pred_bboxes = decode_bboxes(pred_dist, anchor_points, stride_tensor, reg_max)

                    gt = targets[targets[:, 0] == bi]
                    if gt.numel():
                        gt_labels = gt[:, 1].long()
                        gt_bboxes = xywh2xyxy(gt[:, 2:6])
                    else:
                        gt_labels = torch.zeros((0,), device=imgs.device, dtype=torch.long)
                        gt_bboxes = torch.zeros((0, 4), device=imgs.device)

                    target_labels, target_bboxes, target_scores, fg_mask = assigner(
                        pred_scores.detach(), pred_bboxes.detach(), anchor_points, gt_labels, gt_bboxes
                    )

                    loss, items = criterion(
                        pred_dist,
                        pred_obj,
                        pred_cls,
                        anchor_points,
                        stride_tensor,
                        target_bboxes,
                        target_scores,
                        fg_mask,
                    )
                    loss_total += loss
                    for k in logs:
                        logs[k] += float(items[k].item())

                loss_total = loss_total / max(imgs.shape[0], 1)

            scaler.scale(loss_total).backward()

            if (ni + 1) % int(cfg.get("accumulate", 1)) == 0:
                if cfg.get("grad_clip", 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

            pbar.set_postfix({k: f"{v / max(imgs.shape[0], 1):.4f}" for k, v in logs.items()})

        sch.step()

        do_val = ((epoch + 1) % val_interval == 0) or ((epoch + 1) == epochs)
        map50 = -1.0
        if do_val:
            eval_model = ema.ema if ema is not None else model
            print(f"[val] epoch={epoch + 1} start")
            metrics = evaluate(eval_model, val_loader, cfg, device)
            map50 = metrics.get("map_50", 0.0)
            log_line = f"[val] epoch={epoch + 1} {metrics}"
            print(log_line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        else:
            print(f"[val] epoch={epoch + 1} skipped (val_interval={val_interval})")

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "cfg": cfg,
            "data": data,
            "backbone": args.backbone,
        }
        torch.save(ckpt, os.path.join(project, "weights", "last.pt"))
        if do_val and map50 > best:
            best = map50
            torch.save(ckpt, os.path.join(project, "weights", "best.pt"))

    print(f"Done. best={best:.4f}")
    return project
