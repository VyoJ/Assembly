"""Fetch datasets + model weights into data/ so experiments never touch the network.

Layout this produces:
    data/cmapss/{train,test,RUL}_FD001.txt
    data/mvtec/metal_nut/{train,test,ground_truth}/...   (manual, see below)
    data/weights/{yolo11n.pt, yolo11s.pt, ...}

    uv run python scripts/download_data.py                 # everything
    uv run python scripts/download_data.py --only cmapss
    uv run python scripts/download_data.py --only cmapss --cmapss-zip ~/Downloads/turbofan.zip
"""

import argparse
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from core.config import settings

CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CMAPSS_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
YOLO_WEIGHTS = ["yolo11n.pt", "yolo11s.pt"]


def _extract_nested(zip_bytes: bytes, out: Path) -> None:
    """Extract a zip, and any zips inside it, into `out`."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            data = zf.read(name)
            if name.lower().endswith(".zip"):
                _extract_nested(data, out)
            else:
                target = out / Path(name).name
                target.write_bytes(data)


def cmapss(zip_path: str | None = None) -> None:
    dest = settings.data_dir / "cmapss"
    if all((dest / f).exists() for f in CMAPSS_FILES):
        print(f"ok    C-MAPSS already in {dest}")
        return
    if zip_path:
        payload = Path(zip_path).read_bytes()
    else:
        print("get   C-MAPSS (NASA, ~12 MB; the server can be slow)")
        with urllib.request.urlopen(CMAPSS_URL, timeout=1800) as r:
            payload = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _extract_nested(payload, Path(tmp))
        for f in Path(tmp).glob("*_FD00*.txt"):
            shutil.copy(f, dest / f.name)
    missing = [f for f in CMAPSS_FILES if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"C-MAPSS archive did not contain {missing}")
    print(f"ok    C-MAPSS -> {dest}")


def mvtec() -> None:
    dest = settings.data_dir / "mvtec" / "metal_nut"
    if (dest / "train").exists():
        print(f"ok    MVTec metal_nut already in {dest}")
        return
    print(f"TODO  MVTec AD needs the license form at mvtec.com; extract metal_nut into {dest}")


def yolo_weights() -> None:
    dest = settings.data_dir / "weights"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("skip  YOLO weights: ultralytics not installed (uv sync --group ml)")
        return
    for name in YOLO_WEIGHTS:
        if (dest / name).exists():
            print(f"ok    {name}")
            continue
        YOLO(name)  # downloads into the cwd
        shutil.move(name, dest / name)
        print(f"ok    {name} -> {dest}")


STEPS = {"cmapss": cmapss, "mvtec": mvtec, "yolo": yolo_weights}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(STEPS), action="append")
    parser.add_argument("--cmapss-zip", help="use an already downloaded NASA turbofan zip")
    args = parser.parse_args()
    print(f"data dir: {settings.data_dir}")
    for name in args.only or STEPS:
        STEPS[name](args.cmapss_zip) if name == "cmapss" else STEPS[name]()


if __name__ == "__main__":
    main()
