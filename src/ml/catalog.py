"""Plant catalog: which streams and label sources exist, and how to read them.

Sensor streams come back as one row per (unit, cycle) with the value in a column named after
the stream id; `read_streams` joins several of them into one wide table.
Sources (see plant/catalog.yaml):
    cmapss:FD001:<col>     column of data/cmapss/train_FD001.txt (every unit runs to failure)
    synthetic:<name>       distractor generated on the C-MAPSS (unit, cycle) grid, unrelated to failures
    mvtec:metal_nut:*      TODO(ML): image stream + QC labels
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.config import settings
from core.contracts import LabelSource, StreamInfo

CATALOG_PATH = Path(__file__).resolve().parents[1] / "plant" / "catalog.yaml"

CMAPSS_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
KEYS = ["unit", "cycle"]


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


@lru_cache
def _cmapss(subset: str) -> pd.DataFrame:
    path = settings.data_dir / "cmapss" / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/download_data.py")
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def _synthetic(name: str) -> pd.Series:
    """Distractors on the FD001 grid. Seeded, so every call returns the same values."""
    grid = _cmapss("FD001")[KEYS]
    rng = np.random.default_rng(sum(map(ord, name)))
    n, cycle = len(grid), grid["cycle"].to_numpy()
    if name == "ambient_temp":    # degC: daily-ish swing + noise
        return pd.Series(21 + 2.5 * np.sin(2 * np.pi * cycle / 24) + rng.normal(0, 0.6, n))
    if name == "shift_counter":   # 3 shifts of 8 cycles
        return pd.Series((cycle // 8) % 3)
    if name == "noise":           # dB: hall noise level
        return pd.Series(72 + rng.normal(0, 3, n))
    raise KeyError(f"unknown synthetic stream '{name}'")


def read_stream(stream_id: str) -> pd.DataFrame:
    """Sensor stream -> DataFrame(unit, cycle, <stream_id>)."""
    dataset, *rest = source_of(stream_id).split(":")
    if dataset == "cmapss":
        subset, col = rest
        df = _cmapss(subset)[KEYS + [col]]
        return df.rename(columns={col: stream_id}).reset_index(drop=True)
    if dataset == "synthetic":
        df = _cmapss("FD001")[KEYS].copy()
        df[stream_id] = _synthetic(rest[0]).to_numpy()
        return df
    if dataset == "mvtec":
        raise NotImplementedError("TODO(ML): catalog.read_stream for MVTec images")
    raise ValueError(f"unknown source '{dataset}' for stream '{stream_id}'")


def read_streams(stream_ids: list[str]) -> pd.DataFrame:
    """Several sensor streams joined on (unit, cycle): one wide table."""
    df = read_stream(stream_ids[0])
    for sid in stream_ids[1:]:
        df = df.merge(read_stream(sid), on=KEYS, how="outer")
    return df.sort_values(KEYS).reset_index(drop=True)


def read_labels(label_source_id: str) -> pd.DataFrame:
    """maintenance_log -> DataFrame(unit, failure_cycle); qc_inspections -> TODO(ML)."""
    dataset, subset, kind = source_of(label_source_id).split(":")
    if dataset == "cmapss" and kind == "failures":
        df = _cmapss(subset).groupby("unit", as_index=False)["cycle"].max()
        return df.rename(columns={"cycle": "failure_cycle"})
    if dataset == "mvtec":
        raise NotImplementedError("TODO(ML): catalog.read_labels for qc_inspections")
    raise ValueError(f"unknown label source '{label_source_id}'")


def peek_stream(stream_id: str, n: int = 5) -> str:
    """First n rows of a stream as text, plus its shape, for the agent's peek tool."""
    df = read_stream(stream_id)
    head = df.head(n).to_string(index=False)
    return f"{stream_id}: {len(df)} rows, {df['unit'].nunique()} units\n{head}"
