"""Detección de dispositivo. En ROCm-PyTorch la GPU AMD aparece como 'cuda'."""
from __future__ import annotations


def get_device() -> tuple[str, str]:
    """Devuelve (torch_device, etiqueta legible)."""
    import torch

    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        name = torch.cuda.get_device_name(0)
        return "cuda", f"{name} ({'AMD ROCm ' + hip if hip else 'CUDA'})"
    return "cpu", "cpu"
