"""Loader and model registry. Adding new support = registering a piece."""
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
        raise ValueError(f"unsupported modality: {modality!r}. Available: {sorted(LOADERS)}")
    return LOADERS[modality]


def get_model(modality: str, task: str, name: str) -> Callable:
    key = f"{modality}/{task}/{name}"
    if key not in MODELS:
        raise ValueError(f"unsupported combination: {key!r}. Available: {sorted(MODELS)}")
    return MODELS[key]


def list_capabilities() -> dict[str, list[str]]:
    """What the LLM can ask for. Returned as-is as the tool description."""
    return {"modalities": sorted(LOADERS), "models": sorted(MODELS)}
