"""Generic pipeline: loader(modality) -> model(modality/task) -> fit -> eval -> export."""
from __future__ import annotations

from .registry import get_loader, get_model
from .schemas import TrainConfig, TrainResult


def run(data_path: str, config: TrainConfig) -> TrainResult:
    loader = get_loader(config.modality)
    build_model = get_model(config.modality, config.task, config.model)
    # TODO: dataset = loader(data_path, config); model = build_model(config)
    #       model.fit(dataset); metrics = model.evaluate(dataset)
    #       if config.export_onnx: export_onnx(...)
    raise NotImplementedError("pipeline not implemented yet")
