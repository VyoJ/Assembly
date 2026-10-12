"""Funciones públicas que orchestration expone como tools al LLM."""
from __future__ import annotations

import time
import traceback

from . import loaders, models  # noqa: F401  (registran piezas al importar)
from .pipeline import run
from .registry import list_capabilities
from .schemas import TrainConfig, TrainResult


def train_model(data_path: str, config: TrainConfig) -> TrainResult:
    """Entrada única. Nunca lanza: los errores vuelven como TrainResult(status='error')."""
    t0 = time.time()
    try:
        result = run(data_path, config)
    except Exception as e:  # noqa: BLE001 - el agente necesita un error legible
        return TrainResult(
            modality=config.modality, task=config.task, status="error",
            error=f"{type(e).__name__}: {e}",
            log=[traceback.format_exc(limit=3)], duration_s=time.time() - t0,
        )
    result.duration_s = time.time() - t0
    return result


def capabilities() -> dict[str, list[str]]:
    return list_capabilities()


# Atajos para el demo (equivalen a train_model con valores por defecto)
def train_vision_model(data_dir: str, config: TrainConfig) -> TrainResult:
    config.modality = "image"
    return train_model(data_dir, config)


def train_tabular_model(data_path: str, config: TrainConfig) -> TrainResult:
    config.modality = "timeseries"
    return train_model(data_path, config)
