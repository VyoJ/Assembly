"""Plant catalog: which streams and label sources exist, and how to read them.

`list_catalog` is real (reads plant/catalog.yaml). Reading actual data is TODO(ML).
"""

from functools import lru_cache
from pathlib import Path

import yaml

from core.contracts import LabelSource, StreamInfo

CATALOG_PATH = Path(__file__).resolve().parents[1] / "plant" / "catalog.yaml"


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


def read_stream(stream_id: str):
    """TODO(ML): return a DataFrame (sensor streams: unit, cycle, value) or image index (image streams).

    Resolve `source_of(stream_id)`, e.g. 'cmapss:FD001:s7' -> column s7 of data/cmapss/train_FD001.txt.
    """
    raise NotImplementedError("TODO(ML): catalog.read_stream")


def read_labels(label_source_id: str):
    """TODO(ML): maintenance_log -> DataFrame(unit, failure_cycle); qc_inspections -> DataFrame(image, defect_type, mask_path)."""
    raise NotImplementedError("TODO(ML): catalog.read_labels")


def peek_stream(stream_id: str, n: int = 5) -> str:
    """TODO(ML): first n rows / items of a stream as text, for the agent's peek tool."""
    raise NotImplementedError("TODO(ML): catalog.peek_stream")
