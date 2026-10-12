"""Series temporales: ventaneo + features (CWRU .mat, MAFAULDA .csv). TODO."""
from ..registry import register_loader


@register_loader("timeseries")
def load_timeseries(data_path, config):
    raise NotImplementedError
