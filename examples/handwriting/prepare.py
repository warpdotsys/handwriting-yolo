"""Prepare genuine handwriting annotations and separately marked composites for YOLO detection."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import struct
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

ALIASES = {
    r"\times": "×",
    r"\div": "÷",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\theta": "θ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\sum": "∑",
    r"\int": "∫",
    r"\infty": "∞",
    r"\pm": "±",
    r"\sqrt": "√",
    r"\{": "{",
    r"\}": "}",
    r"\ldots": "…",
    r"\cdots": "…",
    r"\cdot": "·",
}


def canonical(label):
    """Map explicit mathematical aliases while preserving case and character identity."""
    label = ALIASES.get(label, label)
    if not label or not label.isprintable() or label.isspace() or any(unicodedata.category(c) == "Co" for c in label):
        return None
    return "latex:" + label if len(label) > 1 else label


def dgrl(data):
    """Restore an original page in its original coordinate frame, combining overlapping foreground."""
    header = struct.unpack_from("<I", data)[0]
    code_length, bits = struct.unpack_from("<HH", data, header - 4)
    if bits != 8:
        raise ValueError("Expected 8-bit DGRL images")
    height, width, line_count = struct.unpack_from("<III", data, header)
    image = np.full((height, width), 255, np.uint8)
    offset = header + 12
    for _ in range(line_count):
        count = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + count * code_length
        top, left, h, w = struct.unpack_from("<4I", data, offset)
        offset += 16
        bitmap = np.frombuffer(data, np.uint8, h * w, offset).reshape(h, w)
        np.minimum(image[top : top + h, left : left + w], bitmap, out=image[top : top + h, left : left + w])
        offset += h * w
    if offset != len(data):
        raise ValueError("Unexpected DGRL record length")
    return Image.fromarray(image).convert("RGB")


class Builder:
    """Own format-specific reading and retain per-image provenance for reproducible preparation."""

    def __init__(self, raw, output, profile, seed):
        self.raw, self.output, self.profile = raw, output, profile
        self.rng = random.Random(seed)
        self.samples, self.skips = [], Counter()
        self.hashes = {"train": set(), "val": set()}
        for split in ("train", "val"):
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    def save(self, image, objects, split, source, key, synthetic=False):
        """Validate original coordinates and record a derivative image without changing raw data."""
        valid = []
        for label, box in objects:
            label = canonical(label)
            if label is None:
                self.skips["invalid_label"] += 1
                continue
            x1, y1, x2, y2 = map(float, box)
            tolerance = max(image.size) * 1e-5
            if not (
                -tolerance <= x1 < x2 <= image.width + tolerance and -tolerance <= y1 < y2 <= image.height + tolerance
            ):
                self.skips["invalid_box"] += 1
                continue
            valid.append((label, [max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)]))
        if not valid:
            return
        image = image.convert("RGB")
        digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        if digest in self.hashes[split] or digest in self.hashes["train" if split == "val" else "val"]:
            self.skips["duplicate_image"] += 1
            return
        self.hashes[split].add(digest)
        stem = source + "_" + hashlib.sha256(str(key).encode()).hexdigest()[:20]
        relative = Path("images") / split / (stem + ".jpg")
        image.save(self.output / relative, quality=92)
        self.samples.append(
            {
                "image": relative.as_posix(),
                "split": split,
                "source": source,
                "source_key": str(key),
                "synthetic_layout": synthetic,
                "pixel_sha256": digest,
                "size": list(image.size),
                "objects": valid,
            }
        )

    def composites(self, pools, source, count):
        """Arrange genuine isolated glyphs, explicitly marking the resulting layouts as synthetic."""
        for split, pool in pools.items():
            if not pool:
                raise ValueError(f"Empty glyph pool: {source}/{split}")
            for index in range(count if split == "train" else max(20, count // 4)):
                image, objects, origins = Image.new("RGB", (512, 512), "white"), [], []
                for cell in range(24):
                    label, glyph, origin = self.rng.choice(pool)
                    glyph = glyph.copy().convert("L")
                    foreground = ImageOps.invert(glyph).point(lambda p: 255 if p > 35 else 0).getbbox()
                    if not foreground:
                        continue
                    glyph = glyph.crop(foreground)
                    glyph.thumbnail((64, 68))
                    x = cell % 6 * 82 + self.rng.randint(5, 15)
                    y = cell // 6 * 115 + self.rng.randint(10, 24)
                    image.paste(glyph.convert("RGB"), (x, y))
                    objects.append((label, [x, y, x + glyph.width, y + glyph.height]))
                    origins.append(origin)
                self.save(image, objects, split, source, f"{split}/{index}/" + ",".join(origins), True)

    def chinese(self):
        """Pair published supplemental boxes with original DGRL pages using official train/test writers."""
        lookup = {}
        for path in (self.raw / "01_简中页面" / "原始文件").rglob("*.dgrl"):
            if path.stem.startswith("C"):
                continue  # Preserve ICDAR2013 as an untouched future test set.
            lookup[path.stem] = path
        labels = json.loads((self.raw / "09_简中字符布局_需对齐" / "ch_temp_train.json").read_text(encoding="utf-8"))
        keys = {"train": [], "val": []}
        for key in labels:
            path = lookup[key]
            split = "train" if "Train" in path.parent.name else "val"
            keys[split].append(key)
        for split, values in keys.items():
            self.rng.shuffle(values)
            limit = (60 if split == "train" else 12) if self.profile == "pilot" else len(values)
            for key in values[:limit]:
                if key in {"088-P16", "007-P19"}:
                    self.skips["known_text_mismatch_page"] += 1
                    continue
                image = dgrl(lookup[key].read_bytes())
                boxes = [obj for line in labels[key]["char_location"] for obj in line]
                if key == "385-P20":
                    boxes = [obj for obj in boxes if obj[1][2] <= image.width]
                candidates = []
                for y in range(0, image.height, 448):
                    for x in range(0, image.width, 448):
                        right, bottom = min(x + 512, image.width), min(y + 512, image.height)
                        selected = []
                        for label, b in boxes:
                            a, c, d, e = b
                            if a < 0 or c < 0 or d > image.width or e > image.height:
                                continue
                            if a < right and d > x and c < bottom and e > y:
                                # Standard detection clipping keeps the original character identity at tile boundaries.
                                clipped = [max(a, x) - x, max(c, y) - y, min(d, right) - x, min(e, bottom) - y]
                                if clipped[2] - clipped[0] >= 2 and clipped[3] - clipped[1] >= 2:
                                    selected.append((label, clipped))
                        if selected and len(selected) >= 3:
                            candidates.append((x, y, right, bottom, selected))
                self.rng.shuffle(candidates)
                for x, y, right, bottom, selected in candidates[:12] if self.profile == "pilot" else candidates:
                    self.save(image.crop((x, y, right, bottom)), selected, split, "casia_page", f"{key}/{x}/{y}")

    def charloc(self):
        """Retain a disjoint writer split and reject malformed normalized character boxes."""
        images = sorted((self.raw / "03_英文字符框" / "原始文件").glob("CharLoc_v3-0/Characters/Writer_*/Images/*.png"))
        grouped = {"train": [], "val": []}
        for image in images:
            writer = int(image.parent.parent.name.split("_")[1])
            grouped["val" if writer % 5 == 0 else "train"].append(image)
        for split, paths in grouped.items():
            self.rng.shuffle(paths)
            limit = (1200 if split == "train" else 250) if self.profile == "pilot" else len(paths)
            for path in paths[:limit]:
                image = Image.open(path).convert("RGB")
                objects = []
                label_file = path.parent.parent / "labels_bb" / (path.stem + ".txt")
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    label, *coords = line.split()
                    xc, yc, width, height = map(float, coords)
                    objects.append(
                        (
                            label,
                            [
                                (xc - width / 2) * image.width,
                                (yc - height / 2) * image.height,
                                (xc + width / 2) * image.width,
                                (yc + height / 2) * image.height,
                            ],
                        )
                    )
                self.save(image, objects, split, "charloc", path.relative_to(self.raw))

    def gnt(self):
        """Read original binary glyph records from disjoint official train/test writer files."""
        pools = {"train": [], "val": []}
        for split, pool in pools.items():
            kind = "Train" if split == "train" else "Test"
            paths = sorted(p for p in (self.raw / "02_简中单字" / "原始文件").rglob("*.gnt") if kind in str(p.parent))
            if self.profile == "full":
                native = self.raw / "02_简中单字" / "原始文件"
                by_archive = {}
                for path in paths:
                    by_archive.setdefault(path.relative_to(native).parts[0], path)
                paths = list(by_archive.values())
            else:
                paths = paths[:4]
            for path in paths:
                data, offset, number = path.read_bytes(), 0, 0
                while offset < len(data):
                    size = struct.unpack_from("<I", data, offset)[0]
                    width, height = struct.unpack_from("<HH", data, offset + 6)
                    if size != 10 + width * height:
                        raise ValueError("Invalid GNT record")
                    if self.profile == "full" or number % 5 == 0:
                        label = data[offset + 4 : offset + 6].rstrip(b"\x00").decode("gb18030")
                        glyph = Image.fromarray(
                            np.frombuffer(data, np.uint8, width * height, offset + 10).reshape(height, width)
                        )
                        pool.append((label, glyph, path.stem + "/" + str(number)))
                    number += 1
                    offset += size
        self.composites(pools, "casia_glyph_composite", 160 if self.profile == "pilot" else 4000)

    def emnist(self):
        """Use the independent official train/test records, correcting the source storage orientation."""
        directory = self.raw / "04_英文和数字单字" / "原始文件" / "gzip"
        mapping = {
            int(a): chr(int(b))
            for a, b in (line.split() for line in (directory / "emnist-byclass-mapping.txt").read_text().splitlines())
        }
        pools = {"train": [], "val": []}
        for split, pool in pools.items():
            kind = "train" if split == "train" else "test"
            with gzip.open(directory / f"emnist-byclass-{kind}-images-idx3-ubyte.gz") as images, gzip.open(
                directory / f"emnist-byclass-{kind}-labels-idx1-ubyte.gz"
            ) as labels:
                magic, count, height, width = struct.unpack(">4I", images.read(16))
                label_magic, label_count = struct.unpack(">2I", labels.read(8))
                if magic != 2051 or label_magic != 2049 or count != label_count:
                    raise ValueError("Invalid EMNIST IDX headers")
                for index in range(min(count, 4000 if self.profile == "pilot" else 24000)):
                    glyph = Image.fromarray(
                        np.frombuffer(images.read(height * width), np.uint8).reshape(height, width).T
                    )
                    pool.append((mapping[labels.read(1)[0]], ImageOps.invert(glyph), f"{kind}/{index}"))
        self.composites(pools, "emnist_composite", 160 if self.profile == "pilot" else 4000)

    def hasy(self):
        """Stream selected genuine symbols and split by user rather than reusing overlapping folds."""
        directory = self.raw / "05_符号单字"
        rows = list(csv.DictReader((directory / "hasy-data-labels.csv").open(encoding="utf-8", newline="")))
        self.rng.shuffle(rows)
        chosen, pools = {}, {"train": [], "val": []}
        counts = Counter()
        for row in rows:
            split = "val" if int(hashlib.sha256(row["user_id"].encode()).hexdigest(), 16) % 5 == 0 else "train"
            if counts[split] < (1200 if self.profile == "pilot" else 12000):
                chosen[row["path"]] = (split, row)
                counts[split] += 1
        with tarfile.open(directory / "HASYv2.tar.bz2", mode="r|bz2") as archive:
            for member in archive:
                if member.name in chosen:
                    split, row = chosen[member.name]
                    glyph = Image.open(io.BytesIO(archive.extractfile(member).read())).convert("L")
                    if np.array(glyph).mean() < 128:
                        glyph = ImageOps.invert(glyph)
                    pools[split].append((row["latex"], glyph, row["path"] + "/user:" + row["user_id"]))
        self.composites(pools, "hasy_composite", 120 if self.profile == "pilot" else 3000)

    def crohme(self):
        """Render original pen trajectories with symbol groups in the very same coordinate transform."""
        ns = "{http://www.w3.org/2003/InkML}"
        with zipfile.ZipFile(self.raw / "06_数学表达式" / "CROHME23.zip") as archive:
            for split, original in (("train", "train"), ("val", "val")):
                names = [
                    n
                    for n in archive.namelist()
                    if f"/INKML/{original}/" in n and n.endswith(".inkml") and "Artificial" not in n
                ]
                self.rng.shuffle(names)
                for name in names[
                    : 350
                    if self.profile == "pilot" and split == "train"
                    else 100
                    if self.profile == "pilot"
                    else len(names)
                ]:
                    root = ET.fromstring(archive.read(name))
                    traces = {}
                    for trace in root.findall(ns + "trace"):
                        points = [
                            tuple(map(float, part.strip().split()[:2]))
                            for part in trace.text.split(",")
                            if part.strip()
                        ]
                        traces[trace.attrib["id"]] = points
                    all_points = [p for points in traces.values() for p in points]
                    if not all_points:
                        self.skips["empty_inkml"] += 1
                        continue
                    x0, y0 = np.min(all_points, axis=0)
                    x1, y1 = np.max(all_points, axis=0)
                    scale = 480 / max(x1 - x0, y1 - y0, 1)

                    def transform(point, x0=x0, y0=y0, scale=scale):
                        return (point[0] - x0) * scale + 16, (point[1] - y0) * scale + 16

                    image = Image.new("RGB", (int((x1 - x0) * scale) + 33, int((y1 - y0) * scale) + 33), "white")
                    draw = ImageDraw.Draw(image)
                    for points in traces.values():
                        converted = [transform(p) for p in points]
                        if len(converted) > 1:
                            draw.line(converted, fill="black", width=3)
                        elif converted:
                            x, y = converted[0]
                            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="black")
                    objects = []
                    for group in root.iter(ns + "traceGroup"):
                        views = group.findall(ns + "traceView")
                        annotation = next(
                            (a for a in group.findall(ns + "annotation") if a.attrib.get("type") == "truth"), None
                        )
                        if not views or annotation is None:
                            continue
                        points = [transform(p) for view in views for p in traces[view.attrib["traceDataRef"]]]
                        low, high = np.min(points, axis=0), np.max(points, axis=0)
                        objects.append(
                            (
                                annotation.text,
                                [
                                    max(0, low[0] - 2),
                                    max(0, low[1] - 2),
                                    min(image.width, high[0] + 2),
                                    min(image.height, high[1] + 2),
                                ],
                            )
                        )
                    self.save(image, objects, split, "crohme_trajectory", name)

    def gnhk(self):
        """Create a distinct word-localization task; do not pretend word boxes are character boxes."""
        out = self.output / "word_localization"
        counts = Counter()
        for split, folder in (("train", "train_data"), ("val", "test_data")):
            for part in ("images", "labels"):
                (out / part / split).mkdir(parents=True, exist_ok=True)
            paths = sorted((self.raw / "07_英文页面_GNHK" / "原始文件" / folder).rglob("*.json"))
            for path in paths[: 64 if self.profile == "pilot" else len(paths)]:
                image = Image.open(path.with_suffix(".jpg"))
                labels = []
                for row in json.loads(path.read_text(encoding="utf-8")):
                    if row["type"] != "H":
                        continue
                    polygon = row["polygon"]
                    xs, ys = [polygon[f"x{i}"] for i in range(4)], [polygon[f"y{i}"] for i in range(4)]
                    a, b, c, d = max(0, min(xs)), max(0, min(ys)), min(image.width, max(xs)), min(image.height, max(ys))
                    if c > a and d > b:
                        labels.append(
                            f"0 {(a + c) / 2 / image.width:.8f} {(b + d) / 2 / image.height:.8f} {(c - a) / image.width:.8f} {(d - b) / image.height:.8f}"
                        )
                if labels:
                    target = out / "images" / split / path.with_suffix(".jpg").name
                    if not target.exists():
                        target.write_bytes(path.with_suffix(".jpg").read_bytes())
                    (out / "labels" / split / path.with_suffix(".txt").name).write_text(
                        "\n".join(labels), encoding="utf-8"
                    )
                    counts[split] += 1
        (out / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": out.as_posix(),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "handwritten_word"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return dict(counts)

    def finish(self):
        """Write standard YOLO labels with one deterministic class dictionary and audit provenance."""
        classes = sorted({label for sample in self.samples for label, box in sample["objects"]})
        ids = {label: index for index, label in enumerate(classes)}
        counts, class_counts = Counter(), Counter()
        for sample in self.samples:
            width, height = sample["size"]
            lines = []
            for label, (a, b, c, d) in sample["objects"]:
                lines.append(
                    f"{ids[label]} {(a + c) / 2 / width:.8f} {(b + d) / 2 / height:.8f} {(c - a) / width:.8f} {(d - b) / height:.8f}"
                )
                class_counts[label] += 1
            path = self.output / sample["image"].replace("images/", "labels/")
            path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[sample["split"] + "/" + sample["source"]] += 1
        if self.hashes["train"] & self.hashes["val"]:
            raise ValueError("Train/validation pixel leakage")
        (self.output / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": self.output.as_posix(),
                    "train": "images/train",
                    "val": "images/val",
                    "names": dict(enumerate(classes)),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (self.output / "provenance.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in self.samples) + "\n", encoding="utf-8"
        )
        word_counts = self.gnhk()
        report = {
            "profile": self.profile,
            "classes": len(classes),
            "class_counts": dict(class_counts),
            "images_by_source": dict(counts),
            "skips": dict(self.skips),
            "word_task_images": word_counts,
            "no_pixel_overlap": True,
            "held_out": "ICDAR2013 pages remain outside preparation",
            "touching_dataset": "CASIA-HWDB-T retained as text/segmentation resource; no invented character boxes or independent test claim",
            "raw_root": str(self.raw),
        }
        (self.output / "preparation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in report.items() if k != "class_counts"}, ensure_ascii=False), flush=True)


def main():
    """Prepare a reproducible local pilot or expanded dataset without touching source archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory to keep image and label provenance reproducible")
    builder = Builder(args.raw_root, args.output, args.profile, args.seed)
    for method in (builder.chinese, builder.charloc, builder.gnt, builder.emnist, builder.hasy, builder.crohme):
        print("PREPARING", method.__name__, flush=True)
        method()
    builder.finish()


if __name__ == "__main__":
    main()
