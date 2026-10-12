"""Window features for sensor streams, computed PER GROUP (e.g. engine unit) to avoid leakage.

TODO(ML): implement.
"""

def window_features(df, group_col: str, window: int, stats=("mean", "std", "slope", "min", "max"), fft_bands=None):
    """Rolling-window features per group; returns a DataFrame aligned to df's rows."""
    raise NotImplementedError("TODO(ML): models.features.window_features")
