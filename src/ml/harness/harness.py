"""Harness: copied READ-ONLY into every task workspace. The only API agent code uses for data I/O.

prepare.py side:  catalog(), read_stream(), read_streams(), read_labels(), write_dataset()
both:             load_signal(), load_image()
train.py side:    task(), load_train(), load_val_inputs(), load_test_inputs(), ids(), groups(),
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
    """The stream as a DataFrame. Table streams: (entity, time, <stream_id>).
    Signal / image streams: one row per recording or image with a `path` (see load_signal / load_image)."""
    from ml.catalog import read_stream as _read

    _allowed(stream_id)
    return _read(stream_id)


def read_streams(stream_ids: list[str]):
    """Several table streams joined on (entity, time) into one wide DataFrame."""
    from ml.catalog import read_streams as _read

    for sid in stream_ids:
        _allowed(sid)
    return _read(stream_ids)


def read_labels(label_source_id: str):
    """events -> DataFrame(entity, event_time); annotations -> DataFrame(record_id, <labels>...)."""
    from ml.catalog import read_labels as _read

    _allowed(label_source_id)
    return _read(label_source_id)


def load_signal(path: str, channel: str | None = None):
    """1-D numpy waveform from a recording (.mat / .npy / .wav / .csv). Works in prepare.py and train.py."""
    from ml.catalog import load_signal as _load

    return _load(path, channel)


def load_image(path: str):
    """Image as a numpy array (HxWxC, or HxW for masks). Works in prepare.py and train.py."""
    from ml.catalog import load_image as _load

    return _load(path)


# On-disk layout (shared with validators, runner and scoring). One row per sample, `id` unique per split.
#   {workspace}/data/train.parquet              id, group?, <inputs>, y
#   {workspace}/data/{val,test}_inputs.parquet  id, group?, <inputs>                 (no labels)
#   {workspace}/data/{split}_X.npy              inputs when X is an N-d array (rows match the parquet)
#   {workspace}/data/files/{split}/<id>.<ext>   inputs when X is files (images, recordings); `path` column
#   {workspace}/data/meta.json                  input kind, label encoding, sizes, columns, split strategy
#   {labels_dir}/{val,test}.parquet             id, y                                (hidden from train.py)
#   {workspace}/data/files_y/train/<id>.<ext>   train label files (e.g. masks)
#   {labels_dir}/files_y/{val,test}/<id>.<ext>  val/test label files                 (hidden from train.py)
# Input kinds: "table" (DataFrame), "array" (numpy, any shape), "files" (paths, or a DataFrame with `path`).
# Labels: scalars as-is; lists / dicts (boxes, ...) as JSON strings; existing file paths (masks) are copied.
# meta.json "label_encoding" says which one per split: "value" | "json" | "file".
# Files are copied under neutral names (<id>.<ext>) so the original path can't leak the label.

DATA_DIR = WORKSPACE / "data"
RESERVED = {"id", "group", "y"}


def _ids(split: str, n: int) -> list[str]:
    return [f"{split}-{i:06d}" for i in range(n)]


def _is_file(v) -> bool:
    return isinstance(v, (str, Path)) and Path(v).suffix != "" and Path(v).is_file()


def _inputs(split: str, X, ids: list[str]):
    """-> (DataFrame with id + inputs, optional N-d array, input kind)."""
    import shutil

    import numpy as np
    import pandas as pd

    if isinstance(X, (list, tuple, pd.Series)) and len(X) and _is_file(X[0] if not isinstance(X, pd.Series) else X.iloc[0]):
        X = pd.DataFrame({"path": list(X)})
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True).copy()
        X.columns = [str(c) for c in X.columns]
        clash = RESERVED & set(X.columns)
        if clash:
            raise ValueError(f"{split}: input columns {sorted(clash)} are reserved; rename them")
        kind = "table"
        if "path" in X.columns:
            kind = "files"
            out = DATA_DIR / "files" / split
            out.mkdir(parents=True, exist_ok=True)
            new = []
            for i, p in zip(ids, X["path"], strict=True):
                target = out / f"{i}{Path(p).suffix}"
                shutil.copy(p, target)
                new.append(str(target.relative_to(DATA_DIR)))
            X["path"] = new
        return pd.concat([pd.DataFrame({"id": ids}), X], axis=1), None, kind
    arr = np.asarray(X)
    if arr.dtype == object:
        raise ValueError(f"{split}: X must be a DataFrame, a numeric array or a list of file paths")
    return pd.DataFrame({"id": ids}), arr, "array"


def _labels(split: str, y, ids: list[str], files_root: Path):
    """-> (list of encoded labels, encoding). Copies label files (masks) under neutral names."""
    import shutil

    import numpy as np

    values = list(y)
    if values and all(v is None or _is_file(v) for v in values) and any(v is not None for v in values):
        out = files_root / split
        out.mkdir(parents=True, exist_ok=True)
        enc = []
        for i, v in zip(ids, values, strict=True):
            if v is None:
                enc.append(None)
                continue
            target = out / f"{i}{Path(v).suffix}"
            shutil.copy(v, target)
            enc.append(str(target.relative_to(files_root.parent)))
        return enc, "file"
    if values and isinstance(values[0], (list, tuple, dict, np.ndarray)):
        return [json.dumps(np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for v in values], "json"
    return values, "value"


def write_dataset(train, val, test, *, groups=None, meta: dict | None = None) -> None:
    """Persist the dataset. Call once, at the end of prepare.py.

    train, val, test: (X, y). X can be a DataFrame (tabular features), a numpy array of any shape
        (windows, spectrograms, image tensors), or file paths (images, recordings; or a DataFrame with a
        `path` column plus metadata). y has one label per row: a number / class, a list or dict (e.g. boxes),
        or a file path (e.g. a mask).
    groups: {"train": [...], "val": [...], "test": [...]} group id per row (e.g. machine or recording),
        so validators can check that no group appears in two splits.
    meta: e.g. {"split": "group" | "temporal", "time_col": "time"}; tells validators which checks apply.
    Val/test labels go to the hidden labels dir; train.py never sees them.
    """
    import numpy as np

    labels_dir = os.environ.get("ASSEMBLY_LABELS_DIR")
    if not labels_dir:
        raise RuntimeError("ASSEMBLY_LABELS_DIR is not set: write_dataset only runs inside prepare.py")
    labels_dir = Path(labels_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    groups = groups or {}

    info = {"sizes": {}, "input_kind": {}, "label_encoding": {}}
    for split, (X, y) in (("train", train), ("val", val), ("test", test)):
        n = len(X)
        if len(y) != n:
            raise ValueError(f"{split}: X has {n} rows but y has {len(y)}")
        ids = _ids(split, n)
        df, arr, kind = _inputs(split, X, ids)
        if groups.get(split) is not None:
            if len(groups[split]) != n:
                raise ValueError(f"{split}: X has {n} rows but groups has {len(groups[split])}")
            df.insert(1, "group", list(groups[split]))
        if arr is not None:
            np.save(DATA_DIR / f"{split}_X.npy", arr)
            info["array_shape"] = list(arr.shape[1:])
        hidden = split != "train"
        enc, info["label_encoding"][split] = _labels(split, y, ids, (labels_dir if hidden else DATA_DIR) / "files_y")
        if hidden:
            import pandas as pd

            df.to_parquet(DATA_DIR / f"{split}_inputs.parquet", index=False)
            pd.DataFrame({"id": ids, "y": enc}).to_parquet(labels_dir / f"{split}.parquet", index=False)
        else:
            df["y"] = enc
            df.to_parquet(DATA_DIR / "train.parquet", index=False)
            info["input_columns"] = [c for c in df.columns if c not in RESERVED]
            info["has_groups"] = "group" in df
        info["sizes"][split], info["input_kind"][split] = n, kind
    (DATA_DIR / "meta.json").write_text(json.dumps({**info, **(meta or {})}, indent=2, default=str))


# ---------------------------------------------------------------- train.py side

def _meta() -> dict:
    path = DATA_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: prepare.py has not written the dataset yet")
    return json.loads(path.read_text())


def _load_inputs(split: str, df):
    """Absolute paths for files; the N-d array for array inputs."""
    import numpy as np

    kind = _meta()["input_kind"][split]
    if kind == "files":
        df["path"] = [str(DATA_DIR / p) for p in df["path"]]
    if kind == "array":
        return np.load(DATA_DIR / f"{split}_X.npy")
    return df


def ids(split: str) -> list[str]:
    """Sample ids of a split, in row order. Predictions must follow this order."""
    import pandas as pd

    name = "train.parquet" if split == "train" else f"{split}_inputs.parquet"
    return pd.read_parquet(DATA_DIR / name, columns=["id"])["id"].tolist()


def groups(split: str) -> list | None:
    """Group id per row of a split (e.g. machine or recording), or None if prepare.py gave none."""
    import pandas as pd

    df = pd.read_parquet(DATA_DIR / ("train.parquet" if split == "train" else f"{split}_inputs.parquet"))
    return df["group"].tolist() if "group" in df else None


def load_train():
    """(X_train, y_train). X is a DataFrame (tables; files with absolute `path`) or a numpy array.
    DataFrames keep `id` and `group`; for arrays use ids("train") / groups("train")."""
    import pandas as pd

    df = pd.read_parquet(DATA_DIR / "train.parquet")
    y = df.pop("y")
    encoding = _meta()["label_encoding"]["train"]
    if encoding == "json":
        y = y.map(lambda v: None if pd.isna(v) else json.loads(v))
    elif encoding == "file":
        y = y.map(lambda v: None if pd.isna(v) else str(DATA_DIR / v))
    return _load_inputs("train", df), y


def load_val_inputs():
    """X_val (no labels), same type as load_train's X. Predictions must keep ids("val") order."""
    import pandas as pd

    return _load_inputs("val", pd.read_parquet(DATA_DIR / "val_inputs.parquet"))


def load_test_inputs():
    """X_test (no labels), same type as load_train's X."""
    import pandas as pd

    return _load_inputs("test", pd.read_parquet(DATA_DIR / "test_inputs.parquet"))


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
