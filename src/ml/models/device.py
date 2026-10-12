"""Device selection: ROCm PyTorch exposes AMD GPUs through the torch.cuda API."""

from core.config import settings


def device() -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
