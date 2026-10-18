"""Leakage-free internal splits. train.py never sees val labels, so it carves its own."""

import numpy as np
import pandas as pd


def internal_split(X, y, groups=None, frac: float = 0.2, by_time: bool = False, seed: int = 0):
    """Return (X_tr, y_tr, X_ho, y_ho) split by group (or time) - never by row when groups exist.

    groups: one group id per row; defaults to X["group"] when X has that column.
    by_time: hold out the last `frac` of rows of every group (rows must be in time order).
             Otherwise a random `frac` of whole groups is held out.
    Without groups, rows are split at random (or by position when by_time).
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    if groups is None and isinstance(X, pd.DataFrame) and "group" in X.columns:
        groups = X["group"]
    rng = np.random.default_rng(seed)

    if groups is None:
        n_ho = max(1, round(frac * len(X)))
        idx = np.arange(len(X)) if by_time else rng.permutation(len(X))
        ho = np.zeros(len(X), dtype=bool)
        ho[idx[-n_ho:]] = True
    else:
        g = pd.Series(np.asarray(groups))
        if by_time:
            pos = g.groupby(g).cumcount()
            size = g.map(g.value_counts())
            ho = (pos >= (size * (1 - frac)).round()).to_numpy()
        else:
            uniq = g.unique()
            n_ho = max(1, round(frac * len(uniq)))
            ho = g.isin(rng.permutation(uniq)[:n_ho]).to_numpy()

    if ho.all() or not ho.any():
        raise ValueError("internal_split produced an empty side; lower frac or check groups")
    return X[~ho], y[~ho], X[ho], y[ho]
