"""Fetch/check the pinned raw archive manifest, using Ultralytics' existing download implementation."""

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

from ultralytics.utils.downloads import safe_download


def main():
    """Verify immutable raw archives and expand only the native formats needed by preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("sources.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = args.raw_root / record["local_path"]
        if not path.exists() and not args.verify_only:
            safe_download(record["url"], file=path.name, dir=path.parent, unzip=False)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Raw archive checksum mismatch: {path}")
        print("VERIFIED", record["local_path"], flush=True)
        if args.verify_only or path.parent.name.startswith("06_"):
            continue
        if path.suffix == ".zip":
            category = path.parent.name[:2]
            native = path.parent / "原始文件"
            with zipfile.ZipFile(path) as archive:
                if category == "07":
                    for member in archive.namelist():
                        with zipfile.ZipFile(io.BytesIO(archive.read(member))) as nested:
                            nested.extractall(native / Path(member).stem)
                else:
                    destination = (
                        native / path.stem
                        if category in {"01", "02", "08"}
                        else path.parent
                        if category == "09"
                        else native
                    )
                    archive.extractall(destination)
        elif path.name == "HASYv2.tar.bz2":
            with tarfile.open(path) as archive:
                for name in ("symbols.csv", "hasy-data-labels.csv"):
                    member = next(m for m in archive if m.name.endswith(name))
                    (path.parent / name).write_bytes(archive.extractfile(member).read())
    print("All 27 pinned archives passed SHA256 and size checks", flush=True)


if __name__ == "__main__":
    main()
