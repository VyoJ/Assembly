"""Images (MVTec AD, camera frames). TODO."""
from ..registry import register_loader


@register_loader("image")
def load_image(data_path, config):
    raise NotImplementedError
