"""Harness: copied READ-ONLY into every task workspace. The only API agent code uses for data I/O.

prepare.py side:  catalog(), read_stream(), read_labels(), write_dataset()
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


# ---------------------------------------------------------------- shared

def task() -> dict:
    """This task's TaskSpec as a dict (prediction_format, metrics, kpis, ...)."""
    return json.loads((WORKSPACE / "task.json").read_text())


# ---------------------------------------------------------------- prepare.py side

def catalog() -> dict:
    """TODO(ML): streams + label sources available to this task (ids, modality, description)."""
    raise NotImplementedError("TODO(ML): harness.catalog")


def read_stream(stream_id: str):
    """TODO(ML): sensor stream -> pandas DataFrame (unit, cycle, value); image stream -> DataFrame (image_id, path)."""
    raise NotImplementedError("TODO(ML): harness.read_stream")


def read_labels(label_source_id: str):
    """TODO(ML): maintenance_log -> DataFrame(unit, failure_cycle); qc_inspections -> DataFrame(image_id, defect_type, mask_path)."""
    raise NotImplementedError("TODO(ML): harness.read_labels")


def write_dataset(train, val, test, *, groups=None, meta: dict | None = None) -> None:
    """TODO(ML): persist the dataset.
    train: (X, y)        -> data/train.*
    val/test: (X, y)     -> X to data/{val,test}_inputs.*, y to the HIDDEN labels dir (ASSEMBLY_LABELS_DIR)
    groups: dict split -> group ids, used by validators for leakage checks
    For images, X may be a list of image paths; y follows the task's prediction_format.
    """
    raise NotImplementedError("TODO(ML): harness.write_dataset")


# ---------------------------------------------------------------- train.py side

def load_train():
    """TODO(ML): (X_train, y_train) as written by prepare.py (+ YOLO data.yaml path for detection tasks)."""
    raise NotImplementedError("TODO(ML): harness.load_train")


def load_val_inputs():
    """TODO(ML): X_val (no labels)."""
    raise NotImplementedError("TODO(ML): harness.load_val_inputs")


def load_test_inputs():
    """TODO(ML): X_test (no labels)."""
    raise NotImplementedError("TODO(ML): harness.load_test_inputs")


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
