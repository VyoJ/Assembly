"""Registro de loaders y modelos. Añadir soporte nuevo = registrar una pieza."""
from __future__ import annotations

from typing import Callable

LOADERS: dict[str, Callable] = {}   # modality -> fn(data_path, config) -> dataset
MODELS: dict[str, Callable] = {}    # "<modality>/<task>/<model>" -> factory(config)


def register_loader(modality: str):
    def deco(fn):
        LOADERS[modality] = fn
        return fn
    return deco


def register_model(modality: str, task: str, name: str):
    def deco(fn):
        MODELS[f"{modality}/{task}/{name}"] = fn
        return fn
    return deco


def get_loader(modality: str) -> Callable:
    if modality not in LOADERS:
        raise ValueError(f"modalidad no soportada: {modality!r}. Disponibles: {sorted(LOADERS)}")
    return LOADERS[modality]


def get_model(modality: str, task: str, name: str) -> Callable:
    key = f"{modality}/{task}/{name}"
    if key not in MODELS:
        raise ValueError(f"combinación no soportada: {key!r}. Disponibles: {sorted(MODELS)}")
    return MODELS[key]


def list_capabilities() -> dict[str, list[str]]:
    """Lo que el LLM puede pedir. Se devuelve tal cual como descripción de la tool."""
    return {"modalities": sorted(LOADERS), "models": sorted(MODELS)}
