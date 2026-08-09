# GMS-YOLO: A Lightweight Improved YOLO11 for Maize Tassel Detection

GMS-YOLO is a lightweight object detection model based on Ultralytics YOLO11,
designed for maize tassel detection. This repository contains all custom modules,
model configuration, and training/inference/benchmark scripts.

## Model Architecture

### Main Model: YOLO11-Ghost-C2GAM-MSECA (`models/GMS-YOLO.yaml`)

**Lightweight modification: GhostNetV3 Backbone + C2GAM + MSECA Head**

- **Backbone (GhostNetV3 style)**: `GhostConv` downsampling + `C3GhostV3` (CSP + `GhostBottleneckV3`)
  - `GhostModuleV3`: primary conv + cheap depthwise feature generation
  - `GhostBottleneckV3`: Expansion -> DW -> SE -> Projection + Shortcut
- **C2GAM** (end of backbone): CSP split + `GAM_Attention` (channel + spatial global attention)
- **Head (MSECA)**: `MSECA` (multi-scale 1D-conv channel attention) inserted after each PAN-FPN fusion
- **Loss**: SIoU (SCYLLA IoU: angle + distance + shape cost)

## Custom Modules (`ultralytics/nn/modules/block.py`)

- `GhostSeLayer` - SE channel attention
- `GhostModuleV3Legacy` - 7-arg GhostModule (used internally by GhostBottleneckV3)
- `GhostBottleneckV3` - Ghost bottleneck (Expansion + DW + SE + Projection + Shortcut)
- `GAM_Attention` / `GAM` - Global Attention Mechanism
- `C2GAM` - CSP Bottleneck + GAM
- `MSECA` / `MSECAAttention` - multi-scale ECA attention
- `C3GhostV3` - CSP + GhostBottleneckV3
- `GhostModuleV3` - 5-arg version (cm, c2, k, light, shortcut)
- `GhostModuleV3S` - stride-downsampling version
- `Conv_BN_HSwish` / `bneck` - MobileNetV3 units

## Loss Function

`ultralytics/utils/metrics.py` adds an **SIoU** branch to `bbox_iou`
(arXiv:2205.12740). `ultralytics/utils/loss.py`'s `BboxLoss` selects the IoU type
via the environment variable `ULTRALYTICS_IOU_LOSS` (`siou`/`ciou`), default `siou`.

## Installation

```bash
pip install -r requirements.txt
# This repository bundles the full modified ultralytics source; run from the repo root.
```

## Training

```bash
# Main model (SIoU, from scratch, 300 epochs)
python scripts/train.py --data data.yaml
```

Training protocol: imgsz=640, batch=8, lr0=0.001, lrf=0.01, AdamW (auto),
epochs=300, configurable seed, deterministic=True.

## Inference

```bash
python scripts/predict.py --weights runs/detect/xxx/weights/best.pt --source ./images
```

## Benchmark (parameters / GFLOPs / FPS)

```bash
python scripts/benchmark.py --weights runs/detect/xxx/weights/best.pt
```

## Experimental Results

Full results are in [results/RESULTS.md](results/RESULTS.md).

| Model | Precision | Recall | mAP50 | mAP50-95 | GFLOPs | Params/M | FPS |
|-------|-----------|--------|-------|----------|--------|----------|-----|
| **GMS-YOLO** | 0.942 | 0.870 | 0.934 | **0.64** | 5.8 | 2.44 | 249.3 |

Evaluation on the maize tassel test set (single class), imgsz=640.
FPS measured on RTX 3060, batch=1.

## Dataset

The dataset is not included in this repository. Data is in YOLO format
(single class: tassel). Configure your paths in `data.yaml`.

## Citation

If you use this code, please cite:

```bibtex
@article{GMSYOLO,
  title   = {GMS-YOLO: A Lightweight Improved Model for Maize Tassel Detection},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

AGPL-3.0 (consistent with Ultralytics).
