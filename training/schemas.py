"""Data contract between orchestration and training (JSON-serializable)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Modality = Literal["image", "timeseries", "audio", "video"]
Task = Literal["classification", "anomaly_detection", "regression", "segmentation"]
Status = Literal["ok", "error"]


@dataclass
class TrainConfig:
    modality: Modality
    task: Task
    model: str = "default"              # registered name, see list_capabilities()
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    output_dir: str = "artifacts"
    export_onnx: bool = True
    target: str | None = None           # target column/label if applicable
    extra: dict[str, Any] = field(default_factory=dict)  # loader/model options


@dataclass
class TrainResult:
    modality: str
    task: str
    status: Status = "ok"
    model_path: str | None = None
    onnx_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    device: str = "cpu"                 # "AMD Instinct MI300X (AMD ROCm 6.x)", etc.
    gpu_stats: dict[str, Any] = field(default_factory=dict)  # from amd-smi
    log: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
