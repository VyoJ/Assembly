"""Imágenes (MVTec AD, frames de cámara). TODO."""
from ..registry import register_loader


@register_loader("image")
def load_image(data_path, config):
    raise NotImplementedError
