"""Export a ONNX (PyTorch / XGBoost). MIGraphX queda fuera: lo decide el equipo."""
from __future__ import annotations


def export_onnx(model, sample_input, out_path: str) -> str:
    raise NotImplementedError
