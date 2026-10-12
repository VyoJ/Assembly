"""Device detection. In ROCm-PyTorch the AMD GPU shows up as 'cuda'."""
from __future__ import annotations


def get_device() -> tuple[str, str]:
    """Returns (torch_device, human-readable label)."""
    import torch

    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        name = torch.cuda.get_device_name(0)
        return "cuda", f"{name} ({'AMD ROCm ' + hip if hip else 'CUDA'})"
    return "cpu", "cpu"
