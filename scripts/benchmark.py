# -*- coding: utf-8 -*-
"""GMS-YOLO benchmark: parameters / GFLOPs / FPS (640x640, batch=1, GPU)
Usage:
    python scripts/benchmark.py --weights runs/detect/xxx/weights/best.pt
"""
import argparse
import time

import torch

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description='GMS-YOLO benchmark')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--iters', type=int, default=100)
    args = ap.parse_args()

    m = YOLO(args.weights)
    model = m.model.to('cuda').eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6

    try:
        from thop import profile
        x = torch.rand(1, 3, args.imgsz, args.imgsz).to('cuda')
        flops, _ = profile(model, inputs=(x,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = float('nan')
        print('(thop not installed, skip GFLOPs)')

    x = torch.rand(1, 3, args.imgsz, args.imgsz).to('cuda')
    with torch.no_grad():
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(args.iters):
                model(x)
        torch.cuda.synchronize()
        fps = args.iters / (time.time() - t0)

    print(f'Parameters: {params:.3f} M')
    print(f'GFLOPs ({args.imgsz}x{args.imgsz}): {gflops:.2f}')
    print(f'FPS (batch=1): {fps:.1f}')
    print(f'Latency per frame: {1000 / fps:.1f} ms')


if __name__ == '__main__':
    main()
