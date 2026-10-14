"""Plant catalog: which streams and label sources exist, and how to read them.

Readers are by data KIND (see the `sources` section of plant/catalog.yaml), not by dataset, so a new
dataset is a config entry. What read_stream returns depends on the kind:

    table / synthetic   DataFrame(entity, time, <stream_id>)      one row per entity and time step
    signal              DataFrame(record_id, entity, path, channel, sample_hz, <record metadata>...)
                        -> load the waveform with load_signal(path, channel)
    image_folder/video  DataFrame(record_id, entity, frame_idx, path, ...)
                        -> load the image with load_image(path)

Label sources return a DataFrame too:
    events       (entity, event_time)                 e.g. failure time per machine
    annotations  (record_id, <label columns>...)      e.g. fault type per recording, defect + mask per image
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.config import settings
from core.contracts import LabelSource, StreamInfo

CATALOG_PATH = Path(__file__).resolve().parents[1] / "plant" / "catalog.yaml"
TABLE_KEYS = ["entity", "time"]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


@lru_cache
def _raw() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


def list_catalog() -> tuple[list[StreamInfo], list[LabelSource]]:
    raw = _raw()
    streams = [StreamInfo(**{k: v for k, v in s.items() if k != "mock_hint"}) for s in raw["streams"]]
    labels = [LabelSource(**{k: v for k, v in s.items() if k != "source"}) for s in raw["label_sources"]]
    return streams, labels


def mock_hint(stream_id: str) -> str | None:
    return next((s.get("mock_hint") for s in _raw()["streams"] if s["id"] == stream_id), None)


def source_of(item_id: str) -> str:
    for s in _raw()["streams"] + _raw()["label_sources"]:
        if s["id"] == item_id:
            return s["source"]
    raise KeyError(item_id)


def _resolve(item_id: str) -> tuple[str, dict, str]:
    """'cmapss_fd001:s7' -> ('cmapss_fd001', <source config>, 's7')."""
    source_id, _, field = source_of(item_id).partition(":")
    cfg = _raw().get("sources", {}).get(source_id)
    if cfg is None:
        raise KeyError(f"'{item_id}' points to unknown source '{source_id}' (see sources in catalog.yaml)")
    return source_id, cfg, field


def _path(cfg: dict) -> Path:
    path = settings.data_dir / cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/download_data.py")
    return path


# ---------------------------------------------------------------- table

@lru_cache
def _table(source_id: str) -> pd.DataFrame:
    cfg = _raw()["sources"][source_id]
    path, fmt = _path(cfg), cfg.get("format", "csv")
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "whitespace":
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cfg.get("columns"))
    else:
        df = pd.read_csv(path, names=cfg.get("columns"), header=None if cfg.get("columns") else "infer")
    df = df.rename(columns={cfg["entity"]: "entity", cfg["time"]: "time"})
    return df.sort_values(TABLE_KEYS).reset_index(drop=True)


def _synthetic(name: str, grid: pd.DataFrame) -> np.ndarray:
    """Distractors. Seeded, so every call returns the same values."""
    rng = np.random.default_rng(sum(map(ord, name)))
    t = grid["time"].to_numpy()
    if name == "ambient_temp":  # degC: daily-ish swing + noise
        return 21 + 2.5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.6, len(t))
    if name == "shift_counter":  # 3 shifts of 8 time steps
        return (t // 8) % 3
    if name == "noise":  # dB
        return 72 + rng.normal(0, 3, len(t))
    raise KeyError(f"unknown synthetic stream '{name}'")


# ---------------------------------------------------------------- signal

def _signal_index(source_id: str, cfg: dict, channel: str, with_labels: bool = False) -> pd.DataFrame:
    root, rows = _path(cfg), []
    hidden = {"file"} | (set() if with_labels else set(cfg.get("label_fields", [])))
    for i, rec in zip(_neutral_order(len(cfg["records"])), cfg["records"], strict=True):
        meta = {k: v for k, v in rec.items() if k not in hidden}
        rows.append({
            "record_id": f"{source_id}/rec{i:04d}",
            "entity": cfg.get("entity", source_id),
            "path": str(root / rec["file"]),
            "channel": channel,
            **meta,
        })
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def _neutral_order(n: int) -> np.ndarray:
    """A fixed shuffle used for record ids, so an id never hints at the file name, folder or label."""
    return np.argsort(np.random.default_rng(0).permutation(n))


def load_signal(path: str, channel: str | None = None) -> np.ndarray:
    """1-D float waveform from a .mat / .npy / .wav / .csv file. `channel` picks a variable/column."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        from scipy.io import loadmat

        mat = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        keys = [k for k in mat if channel is None or k.endswith(_channel_suffix(channel))]
        # some files hold more than one recording (e.g. CWRU 99.mat); prefer the one named after the file
        keys.sort(key=lambda k: path.stem.zfill(3) not in k)
        if not keys:
            raise KeyError(f"channel '{channel}' not in {path.name}; variables: {sorted(mat)}")
        return np.asarray(mat[keys[0]], dtype=float).ravel()
    if suffix == ".npy":
        return np.load(path).astype(float).ravel()
    if suffix == ".wav":
        from scipy.io import wavfile

        _, data = wavfile.read(path)
        data = np.asarray(data, dtype=float)
        return data if data.ndim == 1 else data[:, int(channel or 0)]
    if suffix in (".csv", ".txt"):
        df = pd.read_csv(path, header=None, sep=None, engine="python")
        return df.iloc[:, int(channel or 0)].to_numpy(dtype=float)
    raise ValueError(f"unsupported signal file '{path.name}'")


def _channel_suffix(channel: str) -> str:
    for cfg in _raw().get("sources", {}).values():
        if cfg.get("kind") == "signal" and channel in cfg.get("channels", {}):
            return cfg["channels"][channel]
    return channel


# ---------------------------------------------------------------- images / video

def _image_index(source_id: str, cfg: dict) -> pd.DataFrame:
    """MVTec layout: {train,test}/<label>/*.png (+ ground_truth/<label>/*_mask.png).
    Generic layout: <label>/*.png or a flat folder."""
    root, rows = _path(cfg), []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT and "ground_truth" not in p.parts)
    # The fixed shuffle is both the neutral id and the replay order of the live camera feed (mixes good and
    # defective parts). Folder names (= labels) only come back through read_labels.
    for i, p in zip(_neutral_order(len(files)), files, strict=True):
        rel = p.relative_to(root).parts
        rows.append({
            "record_id": f"{source_id}/img{i:05d}",
            "entity": cfg.get("entity", source_id),
            "frame_idx": int(i),
            "path": str(p),
            "source_split": rel[0] if cfg.get("layout") == "mvtec" else None,
        })
    return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)


def _image_annotations(source_id: str, cfg: dict) -> pd.DataFrame:
    root, idx = _path(cfg), _image_index(source_id, cfg)
    rows = []
    for _, r in idx.iterrows():
        p = Path(r["path"])
        label = p.parent.name
        mask = root / "ground_truth" / label / f"{p.stem}_mask.png"
        rows.append({
            "record_id": r["record_id"],
            "label": label,
            "defective": label != "good",
            "mask_path": str(mask) if mask.exists() else None,
        })
    return pd.DataFrame(rows)


def _video_index(source_id: str, cfg: dict) -> pd.DataFrame:
    """Each video becomes frames (cached as PNG under data/_cache), then an image index."""
    import cv2

    root, rows = _path(cfg), []
    every = int(cfg.get("every_n_frames", 1))
    for video in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        out = settings.data_dir / "_cache" / "frames" / source_id / video.stem
        out.mkdir(parents=True, exist_ok=True)
        cap, i = cv2.VideoCapture(str(video)), 0
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                f = out / f"{i:06d}.png"
                if not f.exists():
                    cv2.imwrite(str(f), frame)
                rows.append({
                    "record_id": f"{source_id}/{video.stem}/{i:06d}", "entity": video.stem,
                    "path": str(f), "frame_idx": i, "time_s": i / fps if fps else None,
                })
            i += 1
        cap.release()
    return pd.DataFrame(rows)


def load_image(path: str) -> np.ndarray:
    """Image as an HxWxC uint8 array (HxW for masks)."""
    from PIL import Image

    return np.asarray(Image.open(path))


# ---------------------------------------------------------------- public API

def read_stream(stream_id: str) -> pd.DataFrame:
    """Return the stream as a DataFrame; the columns depend on the source kind (see module docstring)."""
    source_id, cfg, field = _resolve(stream_id)
    kind = cfg["kind"]
    if kind == "table":
        df = _table(source_id)
        return df[TABLE_KEYS + [field]].rename(columns={field: stream_id})
    if kind == "synthetic":
        grid = _table(cfg["grid"])[TABLE_KEYS].copy()
        grid[stream_id] = _synthetic(field, grid)
        return grid
    if kind == "signal":
        return _signal_index(source_id, cfg, field)
    if kind == "image_folder":
        return _image_index(source_id, cfg)
    if kind == "video":
        return _video_index(source_id, cfg)
    raise ValueError(f"unknown source kind '{kind}' for '{stream_id}'")


def read_streams(stream_ids: list[str]) -> pd.DataFrame:
    """Several table streams joined on (entity, time) into one wide table.
    Record streams (signal / image) are stacked, one block per stream."""
    frames = [read_stream(s) for s in stream_ids]
    if all(set(TABLE_KEYS) <= set(f.columns) and "record_id" not in f for f in frames):
        df = frames[0]
        for f in frames[1:]:
            df = df.merge(f, on=TABLE_KEYS, how="outer")
        return df.sort_values(TABLE_KEYS).reset_index(drop=True)
    return pd.concat([f.assign(stream_id=s) for s, f in zip(stream_ids, frames, strict=True)], ignore_index=True)


def read_labels(label_source_id: str) -> pd.DataFrame:
    """events -> (entity, event_time); annotations -> (record_id, <labels>...)."""
    source_id, cfg, field = _resolve(label_source_id)
    kind = cfg["kind"]
    if kind == "table" and field == "last_time":  # run-to-failure: the last observed time is the failure
        df = _table(source_id).groupby("entity", as_index=False)["time"].max()
        return df.rename(columns={"time": "event_time"})
    if kind == "signal" and field == "records":
        idx = _signal_index(source_id, cfg, channel="", with_labels=True)
        return idx.drop(columns=["entity", "path", "channel"])
    if kind == "image_folder" and field == "annotations":
        return _image_annotations(source_id, cfg)
    raise ValueError(f"don't know how to read labels '{field}' from a '{kind}' source")


def peek_stream(stream_id: str, n: int = 5) -> str:
    """First n rows / records of a stream as text, plus a summary, for the agent's peek tool."""
    df = read_stream(stream_id)
    head = df.head(n)
    if "record_id" not in df:
        summary = f"{len(df)} rows, {df['entity'].nunique()} entities"
    elif "channel" in df:
        sig = load_signal(head["path"].iloc[0], head["channel"].iloc[0])
        summary = (f"{len(df)} recordings; first one: {len(sig)} samples, "
                   f"mean {sig.mean():.4f}, std {sig.std():.4f}, first values {np.round(sig[:5], 4).tolist()}")
        head = head.drop(columns=["path"])
    else:
        img = load_image(head["path"].iloc[0])
        summary = f"{len(df)} images; first one: shape {img.shape}, dtype {img.dtype}"
        head = head.drop(columns=["path"])
    return f"{stream_id}: {summary}\n{head.to_string(index=False)}"
