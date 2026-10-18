"""Deterministic 'what does X actually look like' section for data_card.md (no LLM, no hidden labels).

Appended after the validators' card so the agent sees sample rows, group structure, train-label stats and how the
deployed model is called before it writes any train.py. Only train labels are read (train.py sees them anyway).
"""

import json
from pathlib import Path

import pandas as pd

SAMPLE_ROWS = 5
MAX_COLS = 12


def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.4g}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _table_section(data: Path, meta: dict) -> list[str]:
    X = pd.read_parquet(data / "train_inputs.parquet")
    if "sample_id" in X:
        X = X.set_index("sample_id")
    groups = meta.get("groups", {}).get("train")
    lines = ["## What `X` looks like (train)", ""]
    shown = X.iloc[:SAMPLE_ROWS, :MAX_COLS].reset_index()
    if groups is not None and len(groups) == len(X):
        shown.insert(1, "group (harness.groups, not in X)", groups[:SAMPLE_ROWS])
    lines.append(_md_table(shown))
    if X.shape[1] > MAX_COLS:
        lines.append(f"\n({X.shape[1]} feature columns in total; first {MAX_COLS} shown)")
    lines.append("")
    if groups is not None and len(groups) == len(X):
        per = pd.Series(groups).value_counts()
        lines.append(f"- Rows per group: min {per.min()}, median {int(per.median())}, max {per.max()} "
                     f"({per.size} groups)")
        if per.max() > 1:
            lines.append("- Several rows share a group (e.g. one row per time step of a unit). Each row is "
                         "predicted on its own: X carries no group column and no link to earlier rows.")
    return lines


def _label_section(data: Path) -> list[str]:
    p = data / "train_labels.parquet"
    if not p.exists():
        return []
    y = pd.read_parquet(p)
    y = y["y"] if "y" in y else y.iloc[:, -1]
    lines = ["", "## Train labels (`y`)", ""]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        q = y.quantile([0, 0.25, 0.5, 0.75, 1]).round(3).tolist()
        lines.append(f"- numeric: mean {y.mean():.4g}, std {y.std():.4g}, min/25%/50%/75%/max {q}")
    else:
        counts = y.value_counts().sort_index()
        lines.append("- counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return lines


def overview(ws_path: str | Path) -> str:
    data = Path(ws_path) / "data"
    try:
        meta = json.loads((data / "meta.json").read_text())
    except (OSError, ValueError):
        return ""
    kind = meta.get("input_kind", {}).get("train", "table")
    lines: list[str] = []
    try:
        if kind == "table" and (data / "train_inputs.parquet").exists():
            lines += _table_section(data, meta)
        else:
            lines += ["## What `X` looks like (train)", "",
                      f"- input kind: {kind} (see `program.md`: files -> `path` column, array -> numpy array)"]
        lines += _label_section(data)
    except Exception as e:  # noqa: BLE001 - the overview is a convenience; never fail prepare over it
        lines.append(f"(overview unavailable: {type(e).__name__}: {e})")
    lines += [
        "",
        "## How the deployed model is called",
        "",
        ("- `predict_fn` / the saved model receives `X` rows exactly as `harness.load_val_inputs()` returns them "
        "(the columns above), possibly one row at a time in production. It never sees groups, other rows, or "
        "history."),
        ("- So features that need history (rolling windows, lags, trends per unit) must already be columns of `X` "
        "(built in `prepare.py`). In `train.py`, only use transformations of a single row."),
        "- Use the `run_python` / `read_workspace_file` tools to check anything else (e.g. read `prepare.py`).",
    ]
    return "\n".join(lines)
