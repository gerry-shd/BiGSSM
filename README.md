# BiGSSM YOLOv11 Engineering

Project goal: use YOLOv11 pretrained weights as the default starting point, integrate the BiGSSM module, and provide reproducible training, evaluation, inference, and deployment (ONNX/TorchScript).

## Directory and Entry Scripts

The actual logic scripts are at the repository root: `train.py` / `val.py` / `predict.py` / `export_onnx.py`.

## Installation

Using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

Or using `pip`:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

Optional dependencies (graceful fallback if unavailable):

- `timm` (more backbone options)
- `ultralytics` (YOLOv11 weight loading)
- `mamba-ssm` (BiSS module acceleration)
- `torchmetrics` / `pycocotools` (more complete COCO evaluation)
- `thop` (GFLOPs statistics)

## Data Format (YOLO)

```yaml
# data.yaml
path: /abs/path/to/dataset
train: images/train
val: images/val
# test: images/test  # optional
nc: 3
names: ["cls0", "cls1", "cls2"]
```

Label files: `labels/*/*.txt`, one line per object in `class cx cy w h` normalized format.

## End-to-End Training / Validation / Deployment Flow

1. Train: `train.py`
2. Validate: `val.py`
3. Export for deployment: `export_onnx.py` (optional TorchScript)
4. Inference: `predict.py` or deployment-side runtime (ONNX/TorchScript)

## Script Interface Details (with examples)

**train.py**
Purpose: train a BiGSSM YOLOv11 detector.
Main arguments:
`--data` dataset config `data.yaml`
`--cfg` model/training config `configs/default.yaml`
`--project` output directory, default `runs/exp`
`--device` device, `0`/`1` or `cpu`
`--backbone` `resnet50` or `yolo11`
`--weights` resume training from specified weights
`--yolo_weights` path to YOLOv11 pretrained weights
`--no_pretrained` disable pretrained initialization
`--no_mamba` disable Mamba
`--opts` override config values, e.g. `batch=8 img=768`

Examples:

```bash
# Basic training
python train.py --data data.yaml --cfg configs/default.yaml --project runs/exp --device 0

# Use YOLOv11 pretrained weights
python train.py --data data.yaml --cfg configs/default.yaml --yolo_weights yolo11n.pt --backbone yolo11

# Override config (batch/img/assigner, etc.)
python train.py --data data.yaml --opts batch=8 img=768 assigner=simota
```

**val.py**
Purpose: evaluate a trained model and report metrics such as mAP.
Main arguments:
`--data` dataset config
`--cfg` config file
`--weights` weight file, e.g. `runs/exp/weights/best.pt`
`--device` device
`--backbone` backbone network
`--yolo_weights` YOLOv11 weights (when YOLOv11 backbone is used)
`--no_mamba` disable Mamba
`--opts` override config values

Examples:

```bash
python val.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --device 0

# Override config (e.g., larger resolution)
python val.py --data data.yaml --weights runs/exp/weights/best.pt --opts img=768 batch=8
```

**predict.py**
Purpose: run single-image or batch inference, save visualized results, and optionally output JSON.
Main arguments:
`--data` dataset config
`--cfg` config file
`--weights` weight file
`--source` input image or directory
`--save_dir` output directory
`--save_json` optional JSON output path
`--conf` confidence threshold
`--iou` NMS IoU threshold
`--max_det` maximum detections per image

Examples:

```bash
python predict.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --source assets/images

# Output JSON results
python predict.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --source assets/images --save_json runs/predict.json
```

**export_onnx.py**
Purpose: export ONNX, with optional TorchScript export.
Main arguments:
`--data` dataset config
`--cfg` config file
`--weights` weight file
`--onnx` ONNX export path
`--torchscript` optional TorchScript output path

Examples:

```bash
# Export ONNX
python export_onnx.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --onnx model.onnx

# Export ONNX and TorchScript together
python export_onnx.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --onnx model.onnx --torchscript model.ts
```

## Deployment and Inference Examples

**ONNX Runtime inference example** (optional dependency: `onnxruntime`):

```python
import cv2
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

img = cv2.imread("test.jpg")
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)
img = img.astype(np.float32) / 255.0
img = img[None, ...]

outputs = sess.run(None, {"images": img})
print([o.shape for o in outputs])
```

**TorchScript inference example**:

```python
import torch

model = torch.jit.load("model.ts")
model.eval()

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    y = model(x)
print([t.shape for t in y])
```

## Accuracy and Correctness Assurance

The following measures help ensure correct execution and stable, reproducible accuracy:

- Data consistency checks: ensure `nc` matches `names` in `data.yaml`, verify `labels` and `images` paths match, and perform random visualization checks on annotations.
- Training correctness check: run a small-set overfit test (e.g., 20 images) to confirm loss drops quickly to a reasonable range.
- Evaluation loop closure: after each training run, evaluate with `val.py` for mAP50 / mAP50-95 and compare against a baseline model.
- Runtime stability: fix random seeds, lock versions (`requirements.txt`), and keep input size and preprocessing pipeline consistent.

Accuracy is strongly influenced by dataset size, annotation quality, and training configuration. To reach target metrics, record baseline mAP on a fixed dataset and iteratively improve configuration and data quality.
