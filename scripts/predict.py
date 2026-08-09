# -*- coding: utf-8 -*-
"""GMS-YOLO inference script
Usage:
    python scripts/predict.py --weights runs/detect/xxx/weights/best.pt --source image_or_video_or_dir
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='GMS-YOLO inference')
    ap.add_argument('--weights', required=True, help='path to best.pt')
    ap.add_argument('--source', default='0', help='image / video / directory / camera(0)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--save', action='store_true', default=True)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--name', default='predict')
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        project='runs/predict',
        name=args.name,
    )


if __name__ == '__main__':
    main()
