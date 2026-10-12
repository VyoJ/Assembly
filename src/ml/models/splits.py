"""Leakage-free internal splits. train.py never sees val labels, so it carves its own.

TODO(ML): implement.
"""

def internal_split(X, y, groups=None, frac: float = 0.2, by_time: bool = False, seed: int = 0):
    """Return (X_tr, y_tr, X_ho, y_ho) split by group (or time) - never by row when groups exist."""
    raise NotImplementedError("TODO(ML): models.splits.internal_split")
