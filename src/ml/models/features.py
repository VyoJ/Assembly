"""Window features for sensor streams, computed PER GROUP (e.g. engine unit) to avoid leakage.

Rows must already be in time order within each group (harness datasets are sorted by unit, cycle).
Each feature at row t only looks at rows t-window+1 .. t of the same group: no future data.
"""

import numpy as np
import pandas as pd

SKIP = {"id", "group", "y"}


def _slope(df: pd.DataFrame, cols: list[str], g, window: int) -> pd.DataFrame:
    """Least-squares slope of each column over the window: cov(t, x) / var(t)."""
    t = g.cumcount().astype(float)

    def roll(s: pd.Series) -> pd.Series:
        r = s.groupby(df["_g"], sort=False).rolling(window, min_periods=2).mean()
        return r.reset_index(level=0, drop=True).reindex(df.index)

    mt, mtt = roll(t), roll(t * t)
    var_t = (mtt - mt * mt).replace(0, np.nan)
    out = {}
    for c in cols:
        x = df[c].astype(float)
        out[c] = ((roll(t * x) - mt * roll(x)) / var_t).fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def window_features(
    df,
    group_col: str,
    window: int,
    stats=("mean", "std", "slope", "min", "max"),
    fft_bands=None,
    cols: list[str] | None = None,
):
    """Rolling-window features per group; returns a DataFrame aligned to df's rows.

    cols: numeric columns to summarise (default: every numeric column except group_col, id, group, y).
    stats: any of mean, std, min, max, slope, last, delta (last - first in the window).
    fft_bands: int, optional; adds the energy in that many frequency bands per window (slow, long windows only).
    Output columns are named '<col>__<stat>_w<window>'.
    """
    if cols is None:
        cols = [c for c in df.select_dtypes("number").columns if c not in SKIP | {group_col}]
    work = df[cols].astype(float).copy()
    work["_g"] = df[group_col].to_numpy()
    g = work.groupby("_g", sort=False)
    feats = {}

    rolling = g[cols].rolling(window, min_periods=1)
    for stat in stats:
        if stat in ("mean", "std", "min", "max"):
            r = getattr(rolling, stat)().reset_index(level=0, drop=True)
            if stat == "std":
                r = r.fillna(0.0)
        elif stat == "last":
            r = work[cols]
        elif stat == "delta":
            first = g[cols].shift(window - 1)
            r = work[cols] - first.fillna(g[cols].transform("first"))
        elif stat == "slope":
            r = _slope(work, cols, g, window)
        else:
            raise ValueError(f"unknown stat '{stat}'")
        r = r.reindex(work.index)
        feats.update({f"{c}__{stat}_w{window}": r[c] for c in cols})

    if fft_bands:
        def band_energy(x: np.ndarray) -> np.ndarray:
            spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
            return np.array([b.sum() for b in np.array_split(spec[1:], fft_bands)])

        for c in cols:
            vals = np.zeros((len(work), fft_bands))
            for idx in g.indices.values():
                series = work[c].to_numpy()[idx]
                for j in range(len(idx)):
                    seg = series[max(0, j - window + 1): j + 1]
                    if len(seg) > fft_bands:
                        vals[idx[j]] = band_energy(seg)
            for b in range(fft_bands):
                feats[f"{c}__fft{b}_w{window}"] = vals[:, b]

    return pd.DataFrame(feats, index=df.index)
