# -*- coding: utf-8 -*-
"""GMS-YOLO training script
Usage:
    python scripts/train.py --data data.yaml
Options: --epochs --imgsz --batch --workers --device --seed --name
"""
import argparse
import os

os.environ.setdefault('ULTRALYTICS_IOU_LOSS', 'siou')  # GMS-YOLO uses SIoU loss

from ultralytics import YOLO  # noqa: E402

MODEL = 'models/GMS-YOLO.yaml'


def main():
    ap = argparse.ArgumentParser(description='GMS-YOLO training')
    ap.add_argument('--data', default='data.yaml')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    model = YOLO(MODEL)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        lr0=0.001,
        lrf=0.01,
        resume=False,
        deterministic=True,
        plots=True,
        val=True,
        save=True,
        project='runs/detect',
        name=args.name or f'GMS-YOLO_s{args.seed}',
        exist_ok=True,
    )


if __name__ == '__main__':
    main()
