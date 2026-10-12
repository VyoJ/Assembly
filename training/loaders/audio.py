"""Audio -> espectrogramas (MIMII). TODO."""
from ..registry import register_loader


@register_loader("audio")
def load_audio(data_path, config):
    raise NotImplementedError
