"""Harness: copied READ-ONLY into every task workspace. The only API agent code uses for data I/O.

prepare.py side:  catalog(), read_stream(), read_streams(), read_labels(), write_dataset()
train.py side:    task(), load_train(), load_val_inputs(), load_test_inputs(),
                  time_left(), device(), weights(), save_model(), finish()

Configured by env vars set by ml/runner.py (see its docstring).
TODO(ML): implement the functions marked TODO.
"""

import json
import os
import time
from pathlib import Path

_START = time.monotonic()
WORKSPACE = Path(os.environ.get("ASSEMBLY_WORKSPACE", Path(__file__).resolve().parent))
EXP_DIR = Path(os.environ.get("ASSEMBLY_EXP_DIR", WORKSPACE / "exp_local"))
if os.environ.get("ASSEMBLY_DATA_DIR"):  # core.config reads DATA_DIR
    os.environ.setdefault("DATA_DIR", os.environ["ASSEMBLY_DATA_DIR"])


# ---------------------------------------------------------------- shared

def task() -> dict:
    """This task's TaskSpec as a dict (prediction_format, metrics, kpis, ...)."""
    return json.loads((WORKSPACE / "task.json").read_text())


# ---------------------------------------------------------------- prepare.py side

def _allowed(item_id: str) -> None:
    spec = task()
    if item_id not in spec["input_streams"] + spec["label_sources"]:
        raise KeyError(
            f"'{item_id}' is not part of this task. Streams: {spec['input_streams']}; "
            f"label sources: {spec['label_sources']}"
        )


def catalog() -> dict:
    """Streams + label sources available to this task (ids, modality, description)."""
    from ml.catalog import list_catalog

    spec = task()
    streams, labels = list_catalog()
    return {
        "streams": [s.model_dump() for s in streams if s.id in spec["input_streams"]],
        "label_sources": [s.model_dump() for s in labels if s.id in spec["label_sources"]],
    }


def read_stream(stream_id: str):
    """Sensor stream -> DataFrame(unit, cycle, <stream_id>). Several at once: read_streams([...])."""
    from ml.catalog import read_stream as _read

    _allowed(stream_id)
    return _read(stream_id)


def read_streams(stream_ids: list[str]):
    """Several sensor streams joined on (unit, cycle): one wide DataFrame."""
    from ml.catalog import read_streams as _read

    for sid in stream_ids:
        _allowed(sid)
    return _read(stream_ids)


def read_labels(label_source_id: str):
    """maintenance_log -> DataFrame(unit, failure_cycle)."""
    from ml.catalog import read_labels as _read

    _allowed(label_source_id)
    return _read(label_source_id)


# On-disk layout (shared with validators, runner and scoring):
#   {workspace}/data/train.parquet         id, group, <features...>, y
#   {workspace}/data/{val,test}_inputs.parquet   id, group, <features...>     (no labels)
#   {workspace}/data/meta.json             splits, sizes, feature columns, split strategy
#   {labels_dir}/{val,test}.parquet        id, y                             (hidden from train.py)
# `id` is unique within a split ("val-000042"); predictions must carry the same ids.

DATA_DIR = WORKSPACE / "data"
RESERVED = {"id", "group", "y"}


def _frame(split: str, X, y=None, groups=None):
    import pandas as pd

    X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).add_prefix("x")
    X.columns = [str(c) for c in X.columns]
    clash = RESERVED & set(X.columns)
    if clash:
        raise ValueError(f"{split}: feature columns {sorted(clash)} are reserved; rename them")
    n = len(X)
    if y is not None and len(y) != n:
        raise ValueError(f"{split}: X has {n} rows but y has {len(y)}")
    if groups is not None and len(groups) != n:
        raise ValueError(f"{split}: X has {n} rows but groups has {len(groups)}")
    X = X.reset_index(drop=True)
    head = pd.DataFrame({"id": [f"{split}-{i:06d}" for i in range(n)]})
    if groups is not None:
        head["group"] = list(groups)
    df = pd.concat([head, X], axis=1)
    labels = None if y is None else pd.DataFrame({"id": df["id"], "y": list(y)})
    return df, labels


def write_dataset(train, val, test, *, groups=None, meta: dict | None = None) -> None:
    """Persist the dataset. Call once, at the end of prepare.py.

    train, val, test: (X, y) with X a DataFrame (or array) and y one target per row.
    groups: {"train": [...], "val": [...], "test": [...]} group id per row (e.g. machine unit),
            so validators can check that no group appears in two splits.
    meta:   e.g. {"split": "group" | "temporal", "time_col": "cycle"}; tells validators which checks apply.
    Val/test labels go to the hidden labels dir; train.py never sees them.
    """
    labels_dir = os.environ.get("ASSEMBLY_LABELS_DIR")
    if not labels_dir:
        raise RuntimeError("ASSEMBLY_LABELS_DIR is not set: write_dataset only runs inside prepare.py")
    labels_dir = Path(labels_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    groups = groups or {}

    train_df, train_y = _frame("train", *train, groups=groups.get("train"))
    train_df["y"] = train_y["y"]
    train_df.to_parquet(DATA_DIR / "train.parquet", index=False)
    sizes = {"train": len(train_df)}
    for split, (X, y) in (("val", val), ("test", test)):
        df, labels = _frame(split, X, y, groups=groups.get(split))
        df.to_parquet(DATA_DIR / f"{split}_inputs.parquet", index=False)
        labels.to_parquet(labels_dir / f"{split}.parquet", index=False)
        sizes[split] = len(df)

    features = [c for c in train_df.columns if c not in RESERVED]
    info = {"sizes": sizes, "feature_columns": features, "has_groups": "group" in train_df, **(meta or {})}
    (DATA_DIR / "meta.json").write_text(json.dumps(info, indent=2, default=str))


# ---------------------------------------------------------------- train.py side

def _read(name: str):
    import pandas as pd

    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: prepare.py has not written the dataset yet")
    return pd.read_parquet(path)


def load_train():
    """(X_train, y_train). X keeps `id` and `group` (e.g. the machine unit, useful for per-unit features)."""
    df = _read("train.parquet")
    return df.drop(columns=["y"]), df["y"]


def load_val_inputs():
    """X_val with `id` (+ `group`), no labels. Predictions must keep the same ids and order."""
    return _read("val_inputs.parquet")


def load_test_inputs():
    """X_test with `id` (+ `group`), no labels."""
    return _read("test_inputs.parquet")


def time_left() -> float:
    """Seconds left in this experiment's training budget. Size training loops with it."""
    budget = float(os.environ.get("ASSEMBLY_TIME_BUDGET_S", "120"))
    return max(0.0, budget - (time.monotonic() - _START))


def device() -> str:
    """'cuda' on ROCm/NVIDIA (ROCm PyTorch uses the cuda API), else 'mps' or 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def weights(name: str) -> str:
    """Local path to pre-downloaded weights (e.g. 'yolo11n.pt', 'sam3'). Experiments have no network."""
    path = Path(os.environ.get("ASSEMBLY_WEIGHTS_DIR", "data/weights")) / name
    if not path.exists():
        raise FileNotFoundError(f"weights '{name}' not found in {path.parent}; see data_card.md for what's available")
    return str(path)


def save_model(model, **kwargs) -> None:
    """TODO(ML): export to EXP_DIR/model via ml.models.export.save_model (ONNX / joblib / SAM 3 config)."""
    raise NotImplementedError("TODO(ML): harness.save_model")


def finish(predict_fn, *, threshold: float | None = None) -> None:
    """TODO(ML): run predict_fn on val + test inputs, check output matches task()['prediction_format'],
    measure p50/p95 latency per sample, write predictions + timings to EXP_DIR. The runner scores them."""
    raise NotImplementedError("TODO(ML): harness.finish")
