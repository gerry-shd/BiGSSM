# BiGSSM YOLOv11 Engineering

工程目标：以 YOLOv11 预训练为默认起点，融合 BiGSSM 模块，提供可复现训练、评估、推理与部署（ONNX/TorchScript）。

## 目录与入口脚本

根目录的 `train.py` / `val.py` / `predict.py` / `export_onnx.py` 即为实际逻辑脚本。

## 安装

使用 `uv`：

```bash
uv venv
uv pip install -r requirements.txt
```

或使用 `pip`：

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

可选依赖（自动降级）：

- `timm`（更多骨干）
- `ultralytics`（YOLOv11 权重加载）
- `mamba-ssm`（BiSS 模块加速）
- `torchmetrics` / `pycocotools`（更完整 COCO 评估）
- `thop`（GFLOPs 统计）

## 数据格式（YOLO）

```yaml
# data.yaml
path: /abs/path/to/dataset
train: images/train
val: images/val
# test: images/test  # optional
nc: 3
names: ["cls0", "cls1", "cls2"]
```

标签文件：`labels/*/*.txt`，每行 `class cx cy w h`，归一化。

## 训练 / 验证 / 部署总流程

1. 训练：`train.py`
2. 验证：`val.py`
3. 部署导出：`export_onnx.py`（可选 TorchScript）
4. 推理：`predict.py` 或部署端（ONNX/TorchScript）

## 接口脚本说明（含详细示例）

**train.py**
用途：训练 BiGSSM YOLOv11 检测器。
主要参数：
`--data` 数据集配置 `data.yaml`
`--cfg` 模型与训练配置 `configs/default.yaml`
`--project` 输出目录，默认 `runs/exp`
`--device` 设备，`0`/`1` 或 `cpu`
`--backbone` `resnet50` 或 `yolo11`
`--weights` 从指定权重继续训练
`--yolo_weights` YOLOv11 预训练权重路径
`--no_pretrained` 禁用预训练
`--no_mamba` 禁用 Mamba
`--opts` 覆盖配置，例如 `batch=8 img=768`

示例：

```bash
# 基本训练
python train.py --data data.yaml --cfg configs/default.yaml --project runs/exp --device 0

# 使用 YOLOv11 预训练权重
python train.py --data data.yaml --cfg configs/default.yaml --yolo_weights yolo11n.pt --backbone yolo11

# 覆盖配置（batch/img/assigner 等）
python train.py --data data.yaml --opts batch=8 img=768 assigner=simota
```

**val.py**
用途：评估训练好的模型，输出 mAP 等指标。
主要参数：
`--data` 数据集配置
`--cfg` 配置文件
`--weights` 权重文件，如 `runs/exp/weights/best.pt`
`--device` 设备
`--backbone` 主干网络
`--yolo_weights` YOLOv11 权重（若主干为 YOLOv11）
`--no_mamba` 禁用 Mamba
`--opts` 覆盖配置

示例：

```bash
python val.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --device 0

# 覆盖配置（例如更大分辨率）
python val.py --data data.yaml --weights runs/exp/weights/best.pt --opts img=768 batch=8
```

**predict.py**
用途：单张或批量推理，保存可视化结果，并可选输出 JSON。
主要参数：
`--data` 数据集配置
`--cfg` 配置文件
`--weights` 权重文件
`--source` 输入图片或目录
`--save_dir` 输出目录
`--save_json` 可选 JSON 输出路径
`--conf` 置信度阈值
`--iou` NMS IOU 阈值
`--max_det` 每图最大检测数

示例：

```bash
python predict.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --source assets/images

# 输出 JSON 结果
python predict.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --source assets/images --save_json runs/predict.json
```

**export_onnx.py**
用途：导出 ONNX，可选导出 TorchScript。
主要参数：
`--data` 数据集配置
`--cfg` 配置文件
`--weights` 权重文件
`--onnx` 导出 ONNX 路径
`--torchscript` 可选 TorchScript 输出路径

示例：

```bash
# 导出 ONNX
python export_onnx.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --onnx model.onnx

# 同时导出 TorchScript
python export_onnx.py --data data.yaml --cfg configs/default.yaml --weights runs/exp/weights/best.pt --onnx model.onnx --torchscript model.ts
```

## 部署与推理示例

**ONNX Runtime 推理示例**（可选依赖 `onnxruntime`）：

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

**TorchScript 推理示例**：

```python
import torch

model = torch.jit.load("model.ts")
model.eval()

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    y = model(x)
print([t.shape for t in y])
```

## 准确率与正确运行保障

以下措施用于保证算法正确运行，并稳定获得可复现的准确率：

- 数据一致性检查：`data.yaml` 中 `nc` 与 `names` 一致，`labels` 与 `images` 路径匹配，随机抽样可视化检查标注。
- 训练正确性检查：小样本过拟合测试（例如 20 张图），确保 loss 能快速下降到合理范围。
- 评估闭环：每次训练后用 `val.py` 评估 mAP50 / mAP50-95，并与基线模型对比。
- 运行稳定性：固定随机种子、版本锁定（`requirements.txt`）、统一输入尺寸与前处理流程。

准确率受数据规模、标注质量、训练配置影响较大；若需达到目标指标，应在固定数据集上记录基线 mAP 并逐步优化配置与数据质量。
