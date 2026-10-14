"""Fetch datasets + model weights into data/ so experiments never touch the network.

Layout this produces:
    data/cmapss/{train,test,RUL}_FD001.txt
    data/cwru/<record>.mat                                (files listed in plant/catalog.yaml)
    data/mvtec/metal_nut/{train,test,ground_truth}/...   (manual, see below)
    data/weights/{yolo11n.pt, yolo11s.pt, ...}

    uv run python scripts/download_data.py                 # everything
    uv run python scripts/download_data.py --only cmapss
    uv run python scripts/download_data.py --only cmapss --cmapss-zip ~/Downloads/turbofan.zip
    uv run python scripts/download_data.py --only mvtec --mvtec-tar ~/Downloads/metal_nut.tar.xz
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
CWRU_URL = "https://engineering.case.edu/sites/default/files/{file}"
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


def cwru() -> None:
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "src/plant/catalog.yaml").read_text())
    src = cfg["sources"]["cwru_12k"]
    dest = settings.data_dir / src["path"]
    dest.mkdir(parents=True, exist_ok=True)
    for rec in src["records"]:
        target = dest / rec["file"]
        if target.exists():
            continue
        print(f"get   CWRU {rec['file']}")
        with urllib.request.urlopen(CWRU_URL.format(file=rec["file"]), timeout=300) as r:
            data = r.read()
        if not data.startswith(b"MATLAB"):
            raise RuntimeError(f"CWRU {rec['file']}: unexpected response (not a .mat file)")
        target.write_bytes(data)
    print(f"ok    CWRU ({len(src['records'])} recordings) -> {dest}")


def mvtec(tar_path: str | None = None) -> None:
    """MVTec AD needs the license form at mvtec.com; pass the downloaded metal_nut.tar.xz with --mvtec-tar."""
    import tarfile

    dest = settings.data_dir / "mvtec" / "metal_nut"
    if (dest / "train").exists():
        print(f"ok    MVTec metal_nut already in {dest}")
        return
    if not tar_path:
        print(f"TODO  MVTec AD: accept the license at mvtec.com, then --mvtec-tar <metal_nut.tar.xz> (-> {dest})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(tar_path).expanduser()) as tf:
        tf.extractall(dest.parent, filter="data")  # "data" filter rejects absolute paths and ../
    n = sum(1 for _ in dest.rglob("*.png"))
    print(f"ok    MVTec metal_nut ({n} png) -> {dest}")


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


STEPS = {"cmapss": cmapss, "cwru": cwru, "mvtec": mvtec, "yolo": yolo_weights}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(STEPS), action="append")
    parser.add_argument("--cmapss-zip", help="use an already downloaded NASA turbofan zip")
    parser.add_argument("--mvtec-tar", help="the metal_nut.tar.xz downloaded from mvtec.com")
    args = parser.parse_args()
    print(f"data dir: {settings.data_dir}")
    for name in args.only or STEPS:
        if name == "cmapss":
            STEPS[name](args.cmapss_zip)
        elif name == "mvtec":
            STEPS[name](args.mvtec_tar)
        else:
            STEPS[name]()


if __name__ == "__main__":
    main()
