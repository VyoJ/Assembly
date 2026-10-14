"""Read-only sensor quality profiling during discovery, before dataset preparation.

No targets, splits or labels are constructed or read here. ``relevance`` is the
fraction of readings that are finite, not evidence of predictive relevance.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from core.contracts import StreamInfo, StreamProfile, TaskSpec
from ml import catalog


def _sensor_values(frame: pd.DataFrame, stream_id: str) -> pd.Series:
    """Validate the loader's table and select its named sensor (or legacy value)."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("read_stream must return a pandas DataFrame")
    if frame.columns.nlevels != 1 or not frame.columns.is_unique:
        raise ValueError("sensor columns must be flat and unique")
    keys = ["entity", "time"] if {"entity", "time"} <= set(frame.columns) else ["unit", "cycle"]
    if not set(keys) <= set(frame.columns):
        raise ValueError("missing required entity/time or legacy unit/cycle columns")
    if stream_id in keys:
        raise ValueError(f"stream_id must not be a {'/'.join(keys)} key")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{'/'.join(keys)} keys must be non-missing and unique")
    if stream_id in frame.columns:
        return frame[stream_id]
    if keys == ["unit", "cycle"] and set(frame.columns) == {*keys, "value"}:
        return frame["value"]
    raise ValueError(f"missing sensor column '{stream_id}' (legacy value requires only unit/cycle/value)")


def _numeric_stats(values: pd.Series) -> dict:
    """Summarize real numeric readings without coercion or changes to the source.

    Missing and infinite values remain counted, but moments use finite readings
    only. Standard deviation is population (ddof=0); zero for one reading does
    not establish constancy. Empty/all-missing columns need no numeric dtype.
    """
    total = len(values)
    missing = int(values.isna().sum())
    if missing < total and (
        not is_numeric_dtype(values.dtype)
        or is_bool_dtype(values.dtype)
        or is_complex_dtype(values.dtype)
    ):
        raise ValueError(f"expected real numeric sensor dtype, got {values.dtype}; no coercion performed")

    present = values[values.notna()]
    numbers = present.to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(numbers)
    finite = numbers[finite_mask]
    count = len(finite)
    stats = {
        "total": total,
        "missing": missing,
        "missing_percent": 100.0 * missing / total if total else 0.0,
        "infinite": int(np.isinf(numbers).sum()),
        "non_finite": total - count,  # includes missing values
        "finite": count,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "exactly_constant": False,
        "approximately_constant": False,
    }
    if count:
        low, high = float(finite.min()), float(finite.max())
        scale = max(abs(low), abs(high))
        # Scaling avoids overflow/underflow and makes the tolerance invariant to multiplicative unit changes.
        normalized = finite / scale if scale else finite
        midpoint = low / 2 + high / 2
        centered = (finite - midpoint) / scale if scale else finite
        # shortcut: relative span is a heuristic; use sensor resolution when calibration metadata exists.
        nearly_constant = count >= 2 and float(normalized.max() - normalized.min()) <= 1e-6
        stats.update(
            mean=float(midpoint + centered.mean() * scale),
            std=float(centered.std(ddof=0) * scale),
            min=low,
            max=high,
            exactly_constant=count >= 2 and present.iloc[finite_mask].nunique() == 1,
            approximately_constant=nearly_constant,
        )
    return stats


def _stats_reason(stats: dict) -> str:
    """Render statistics in the existing contract's sole explanatory field."""
    moments = "; ".join(
        f"{key}={stats[key]:.6g}" if stats[key] is not None else f"{key}=unavailable"
        for key in ("mean", "std", "min", "max")
    )
    return (
        f"observations={stats['total']}; missing={stats['missing']} ({stats['missing_percent']:.1f}%); "
        f"infinite={stats['infinite']}; non_finite={stats['non_finite']} (includes missing); "
        f"finite={stats['finite']}; {moments} (population std); "
        f"exactly_constant={stats['exactly_constant']}; "
        f"approximately_constant={stats['approximately_constant']}"
    )


def profile_streams(spec: TaskSpec, streams: list[StreamInfo]) -> list[StreamProfile]:
    """Profile every candidate in order, independently of task targets/labels.

    ``spec`` preserves the shared API but does not authorize any label access.
    For valid numeric tables, relevance = finite / total (0 for empty tables),
    including constant signals: quality and variability are separate decisions.
    Unsupported/unavailable inputs use 0 as an explicitly unassessed fallback
    and are retained. Malformed, empty, all-invalid and (near-)constant numeric
    signals are dropped. One finite reading is retained with a warning.
    """
    profiles = []
    for stream in streams:
        relevance, keep = 0.0, True
        if stream.modality not in ("temperature", "pressure", "vibration", "plc", "other"):
            reason = (
                "Detailed profiling is not yet implemented for this modality; retained; quality unassessed."
            )
        else:
            try:
                frame = catalog.read_stream(stream.id)
            except (NotImplementedError, OSError, KeyError, ImportError) as exc:
                reason = f"Readings unavailable ({type(exc).__name__}: {exc}); retained; quality unassessed."
            except ValueError as exc:
                keep = False
                reason = f"Invalid stream source: {exc}."
            else:
                try:
                    if isinstance(frame, pd.DataFrame) and {"record_id", "entity", "path"} <= set(frame):
                        if (not frame.columns.is_unique or frame.columns.nlevels != 1
                                or frame[["record_id", "entity", "path"]].isna().any().any()
                                or frame["record_id"].duplicated().any()):
                            raise ValueError("recording identity and paths must be non-missing and unique")
                        profiles.append(StreamProfile(
                            stream_id=stream.id, relevance=0.0, keep=True,
                            reason=(f"{stream.modality}: {len(frame)} file recordings; retained; quality unassessed. "
                                    "Waveforms are not profiled; no supervised labels are read."),
                        ))
                        continue
                    stats = _numeric_stats(_sensor_values(frame, stream.id))
                except (TypeError, ValueError) as exc:
                    keep = False
                    reason = f"Invalid sensor table: {exc}."
                else:
                    relevance = stats["finite"] / stats["total"] if stats["total"] else 0.0
                    if not stats["finite"]:
                        keep, decision = False, "No usable finite observations."
                    elif stats["exactly_constant"]:
                        keep, decision = False, "Exactly constant finite readings."
                    elif stats["approximately_constant"]:
                        keep = False
                        decision = "Approximately constant finite readings (relative span <= 1e-6)."
                    elif stats["finite"] == 1:
                        decision = (
                            "Only one finite observation; retained, "
                            "insufficient evidence to assess constancy."
                        )
                    else:
                        decision = "Varying finite readings; retained."
                    reason = f"{decision} {_stats_reason(stats)}."
        profiles.append(StreamProfile(
            stream_id=stream.id,
            relevance=relevance,
            keep=keep,
            reason=(
                f"{stream.modality} (unit={stream.unit or 'unspecified'}): {reason} "
                "relevance is finite/total when assessed, otherwise 0 (unassessed). "
                "Predictive relevance is unassessed; no supervised labels are read."
            ),
        ))
    return profiles
