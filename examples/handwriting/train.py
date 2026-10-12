"""Run the local handwriting pilot using the existing Ultralytics training and validation engine."""

import argparse
import json
from pathlib import Path

import torch

from ultralytics import YOLO


def main():
    """Train a small detector with character-safe augmentations and write measured validation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA GPU is required for this local training profile")
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        device=0,
        workers=0,
        project=str(args.runs),
        name="handwriting_pilot",
        seed=20261012,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        max_det=600,
        fliplr=0,
        flipud=0,
        degrees=0,
        shear=0,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        close_mosaic=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.1,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=1,
        cos_lr=True,
        patience=10,
    )
    best = model.trainer.best
    validation = YOLO(str(best)).val(
        data=str(args.data),
        imgsz=512,
        batch=args.batch,
        device=0,
        workers=0,
        max_det=600,
        project=str(args.runs),
        name="pilot_validation",
        plots=True,
    )
    report = {
        "checkpoint": str(best),
        "epochs": args.epochs,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "metrics": validation.results_dict,
        "scope": "First pilot only; sparse classes and synthetic layouts do not establish complete recognition quality",
    }
    (args.runs / "pilot_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
