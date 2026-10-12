"""Detect characters on a full page with overlapping tiles and global duplicate suppression."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def main():
    """Run existing prediction on page tiles and preserve the original page coordinate system."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()
    model, image = YOLO(str(args.model)), Image.open(args.image).convert("RGB")
    detections = []
    for top in range(0, image.height, 448):
        for left in range(0, image.width, 448):
            tile = image.crop((left, top, min(left + 512, image.width), min(top + 512, image.height)))
            if not ImageOps.invert(tile.convert("L")).getbbox():
                continue
            result = model.predict(tile, imgsz=512, device=0, quantize=16, conf=args.conf, max_det=600, verbose=False)[
                0
            ]
            for box, confidence, category in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
            ):
                a, b, c, d = box
                detections.append([a + left, b + top, c + left, d + top, confidence, int(category)])
    selected = []
    if detections:
        values = torch.tensor(detections, dtype=torch.float32)
        # Duplicate predictions across tiles may disagree about class; keep the highest-confidence geometry.
        keep = nms(values[:, :4], values[:, 4], 0.5).tolist()
        selected = [
            {
                "box_xyxy": detections[i][:4],
                "confidence": detections[i][4],
                "class_id": detections[i][5],
                "label": model.names[detections[i][5]],
            }
            for i in keep
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "image": str(args.image),
        "checkpoint": str(args.model),
        "page_size": list(image.size),
        "tile_size": 512,
        "stride": 448,
        "confidence_threshold": args.conf,
        "detections": selected,
        "note": "Character/symbol-group boxes only; no reading-order or mathematical-structure inference",
    }
    args.output.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    annotator = Annotator(np.asarray(image).copy(), line_width=2, font_size=14, pil=True, example="汉字")
    for item in selected:
        annotator.box_label(item["box_xyxy"], f"{item['label']} {item['confidence']:.2f}", color=(230, 65, 40))
    Image.fromarray(annotator.result()).save(args.output.with_suffix(".jpg"), quality=90)
    print(json.dumps({"detections": len(selected), "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
