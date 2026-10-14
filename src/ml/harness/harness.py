"""Harness: copied READ-ONLY into every task workspace. The only API agent code uses for data I/O.

prepare.py side:  catalog(), read_stream(), read_streams(), read_labels(), write_dataset()
both:             load_signal(), load_image()
train.py side:    task(), load_train(), load_val_inputs(), load_test_inputs(), ids(), groups(), classes(),
                  time_left(), device(), weights(), save_model(), finish()

Configured by env vars set by ml/runner.py (see its docstring).
TODO(ML): implement the functions marked TODO.
"""

import json
import math
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


# On-disk layout: ML_PLAN.md "The seam", plus two extensions marked (+). Every sample has a `sample_id`;
# predictions are matched to labels by id, never by row order.
#   data/{split}_inputs.parquet     sample_id + feature columns   (files: sample_id, path [+ metadata])
#   data/{split}_inputs.npy     (+) inputs when X is an N-d array (windows, spectrograms, image tensors);
#                                   the parquet then holds sample_id only, in the same order
#   data/train_labels.parquet       sample_id, y                  (boxes / masks: train_labels.json)
#   data/files/{split}/<id>.<ext> (+) input files copied under neutral names, so a path can't leak the label
#   data/meta.json                  prediction_format, classes?, groups {split: [group per sample]},
#                                   input_kind {split: table|files|array} (+), anything prepare.py passes
#   {labels_dir}/{val,test}_labels.parquet   hidden from train.py  (boxes / masks: {split}_labels.json)
# Labels: scalar / binary_score -> number; class_probs -> class index (class names in meta["classes"]);
# boxes -> {sample_id: [[x1, y1, x2, y2, class], ...]}; masks -> {sample_id: mask_png_path} (copied).

DATA_DIR = WORKSPACE / "data"
RESERVED = {"sample_id", "y"}


def _ids(split: str, n: int) -> list[str]:
    return [f"{split}-{i:06d}" for i in range(n)]


def _is_file(v) -> bool:
    return isinstance(v, (str, Path)) and Path(v).suffix != "" and Path(v).is_file()


def _copy_files(paths, ids: list[str], out: Path) -> list[str | None]:
    """Copy files to out/<sample_id>.<ext>; returns the new absolute paths (None stays None)."""
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    new = []
    for sid, p in zip(ids, paths, strict=True):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            new.append(None)
            continue
        target = out / f"{sid}{Path(p).suffix}"
        shutil.copy(p, target)
        new.append(str(target.resolve()))
    return new


def _write_inputs(split: str, X, ids: list[str]) -> str:
    """Write data/{split}_inputs.parquet (+ .npy for arrays). Returns the input kind."""
    import numpy as np
    import pandas as pd

    if isinstance(X, (list, tuple, pd.Series)) and len(X) and _is_file(next(iter(X))):
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
            X["path"] = [str(Path(p).relative_to(DATA_DIR)) for p in _copy_files(X["path"], ids, DATA_DIR / "files" / split)]
        X.insert(0, "sample_id", ids)
        X.to_parquet(DATA_DIR / f"{split}_inputs.parquet", index=False)
        return kind
    arr = np.asarray(X)
    if arr.dtype == object:
        raise ValueError(f"{split}: X must be a DataFrame, a numeric array or a list of file paths")
    pd.DataFrame({"sample_id": ids}).to_parquet(DATA_DIR / f"{split}_inputs.parquet", index=False)
    np.save(DATA_DIR / f"{split}_inputs.npy", arr)
    return "array"


def _write_labels(split: str, y, ids: list[str], out_dir: Path, fmt: str, classes: list | None) -> None:
    """{split}_labels.parquet (sample_id, y) or {split}_labels.json for boxes / masks."""
    import numpy as np
    import pandas as pd

    values = list(y)
    if fmt == "masks" or (values and all(v is None or _is_file(v) for v in values) and any(v is not None for v in values)):
        masks = _copy_files(values, ids, out_dir / "files_y" / split)
        (out_dir / f"{split}_labels.json").write_text(json.dumps(dict(zip(ids, masks, strict=True))))
        return
    if fmt == "boxes" or (values and isinstance(values[0], (list, tuple, np.ndarray))):
        boxes = [np.asarray(v).tolist() if isinstance(v, np.ndarray) else list(v or []) for v in values]
        (out_dir / f"{split}_labels.json").write_text(json.dumps(dict(zip(ids, boxes, strict=True))))
        return
    if classes is not None:
        lookup = {c: i for i, c in enumerate(classes)}
        values = [lookup[v] for v in values]
    pd.DataFrame({"sample_id": ids, "y": values}).to_parquet(out_dir / f"{split}_labels.parquet", index=False)


def write_dataset(train, val, test, *, groups=None, meta: dict | None = None) -> None:
    """Persist the dataset. Call once, at the end of prepare.py.

    train, val, test: (X, y). X can be a DataFrame (tabular features), a numpy array of any shape
        (windows, spectrograms, image tensors), or file paths (images, recordings; or a DataFrame with a
        `path` column plus metadata). y has one label per row: a number, a class (name or index), a list of
        boxes [[x1, y1, x2, y2, class], ...], or a mask file path.
    groups: {"train": [...], "val": [...], "test": [...]} group id per row (e.g. machine or recording),
        so validators can check that no group appears in two splits.
    meta: e.g. {"split": "group" | "temporal", "time_col": "time", "early_warning_horizon": 30};
        stored in data/meta.json for validators and scoring.
    Val/test labels go to the hidden labels dir; train.py never sees them.
    """
    labels_dir = os.environ.get("ASSEMBLY_LABELS_DIR")
    if not labels_dir:
        raise RuntimeError("ASSEMBLY_LABELS_DIR is not set: write_dataset only runs inside prepare.py")
    labels_dir = Path(labels_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    fmt = task().get("prediction_format")
    splits = {"train": train, "val": val, "test": test}

    classes = None
    if fmt == "class_probs":  # class names -> indices; the names are kept in meta["classes"]
        all_y = [v for _, y in splits.values() for v in y]
        classes = sorted(set(all_y), key=lambda c: (str(type(c)), c))

    info = {"prediction_format": fmt, "input_kind": {}, "sizes": {}, "groups": {}}
    for split, (X, y) in splits.items():
        n = len(X)
        if len(y) != n:
            raise ValueError(f"{split}: X has {n} rows but y has {len(y)}")
        ids = _ids(split, n)
        info["input_kind"][split] = _write_inputs(split, X, ids)
        _write_labels(split, y, ids, DATA_DIR if split == "train" else labels_dir, fmt, classes)
        if groups and groups.get(split) is not None:
            if len(groups[split]) != n:
                raise ValueError(f"{split}: X has {n} rows but groups has {len(groups[split])}")
            info["groups"][split] = [g.item() if hasattr(g, "item") else g for g in groups[split]]
        info["sizes"][split] = n
    if classes is not None:
        info["classes"] = classes
    if not info["groups"]:
        del info["groups"]
    (DATA_DIR / "meta.json").write_text(json.dumps({**info, **(meta or {})}, indent=2, default=str))


# ---------------------------------------------------------------- train.py side

def _meta() -> dict:
    path = DATA_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: prepare.py has not written the dataset yet")
    return json.loads(path.read_text())


def _load_inputs(split: str):
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(DATA_DIR / f"{split}_inputs.parquet").set_index("sample_id")
    kind = _meta()["input_kind"][split]
    if kind == "array":
        return np.load(DATA_DIR / f"{split}_inputs.npy")
    if kind == "files":
        df["path"] = [str(DATA_DIR / p) for p in df["path"]]
    return df


def ids(split: str) -> list[str]:
    """sample_ids of a split, in row order (for array inputs, where X has no index)."""
    import pandas as pd

    return pd.read_parquet(DATA_DIR / f"{split}_inputs.parquet", columns=["sample_id"])["sample_id"].tolist()


def groups(split: str = "train"):
    """Group id per sample (e.g. engine unit or recording) as a Series indexed by sample_id, or None."""
    import pandas as pd

    g = _meta().get("groups", {}).get(split)
    return None if g is None else pd.Series(g, index=pd.Index(ids(split), name="sample_id"), name="group")


def classes() -> list | None:
    """Class names for class_probs tasks: label i means classes()[i]; probability column i too."""
    return _meta().get("classes")


def load_train():
    """(X, y). X: DataFrame indexed by sample_id (tables; files with an absolute `path` column) or a numpy
    array (array inputs; ids("train") gives the matching sample_ids). y: Series indexed by sample_id
    (boxes / masks: lists of boxes / mask paths)."""
    import pandas as pd

    X = _load_inputs("train")
    labels_json = DATA_DIR / "train_labels.json"
    if labels_json.exists():
        data = json.loads(labels_json.read_text())
        y = pd.Series(data, name="y")
    else:
        y = pd.read_parquet(DATA_DIR / "train_labels.parquet").set_index("sample_id")["y"]
    y.index.name = "sample_id"
    return X, y.loc[ids("train")]


def load_val_inputs():
    """X_val (no labels), same type as load_train's X."""
    return _load_inputs("val")


def load_test_inputs():
    """X_test (no labels), same type as load_train's X."""
    return _load_inputs("test")


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
