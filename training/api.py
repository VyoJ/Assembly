"""Public functions that orchestration exposes as tools to the LLM."""
from __future__ import annotations

import time
import traceback

from . import loaders, models  # noqa: F401  (register pieces on import)
from .pipeline import run
from .registry import list_capabilities
from .schemas import TrainConfig, TrainResult


def train_model(data_path: str, config: TrainConfig) -> TrainResult:
    """Single entry point. Never raises: errors come back as TrainResult(status='error')."""
    t0 = time.time()
    try:
        result = run(data_path, config)
    except Exception as e:  # noqa: BLE001 - the agent needs a readable error
        return TrainResult(
            modality=config.modality, task=config.task, status="error",
            error=f"{type(e).__name__}: {e}",
            log=[traceback.format_exc(limit=3)], duration_s=time.time() - t0,
        )
    result.duration_s = time.time() - t0
    return result


def capabilities() -> dict[str, list[str]]:
    return list_capabilities()


# Demo shortcuts (equivalent to train_model with default values)
def train_vision_model(data_dir: str, config: TrainConfig) -> TrainResult:
    config.modality = "image"
    return train_model(data_dir, config)


def train_tabular_model(data_path: str, config: TrainConfig) -> TrainResult:
    config.modality = "timeseries"
    return train_model(data_path, config)
