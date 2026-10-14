"""Read-only validation of the persisted sensor datasets produced by prepare.py.

Supports scalar regression and binary ground truth encoded as numeric 0/1 (or
booleans). Split strategy comes from data/meta.json, never guessed from prose.
Hidden labels are used internally and are never included in diagnostics.
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.types import (
    infer_dtype,
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from core.contracts import ValidationReport, Workspace
from ml import workspace

CHECKS = [
    "files_present",  # data/ has train inputs+labels, val/test inputs; hidden dir has val/test labels
    "format",  # label shape/type matches spec.prediction_format
    "group_disjoint",  # no group (engine unit, image id...) appears in more than one split
    "time_ordered",  # if split_strategy is by time: max(train time) < min(val time) < min(test time)
    "min_sizes",  # each split above a minimum size
    "label_sanity",  # val/test not single-class; enough positives for binary/boxes
    "no_label_leak",  # no input column identical/near-identical to the label
]


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"id", "group", "y"}]


def _finite_numeric(values: pd.Series) -> bool:
    """Real numeric values only; no string conversion or imputation."""
    return bool(
        is_numeric_dtype(values.dtype)
        and not is_complex_dtype(values.dtype)
        and np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan)).all()
    )


def _id_errors(frame: pd.DataFrame, name: str) -> list[str]:
    if "id" not in frame:
        return [f"{name}: missing id column."]
    ids = frame["id"]
    errors = []
    if ids.isna().any():
        errors.append(f"{name}: IDs must be non-null.")
    kind = infer_dtype(ids, skipna=True)
    if kind not in {"string", "integer", "empty"}:
        errors.append(f"{name}: IDs must be strings or integers.")
    elif ids.duplicated().any():
        errors.append(f"{name}: IDs must be unique within the split.")
    if kind == "string" and ids.str.strip().eq("").any():
        errors.append(f"{name}: IDs must not be blank.")
    return errors


def _format_errors(inputs: dict, labels: dict, meta: dict) -> list[str]:
    errors = []
    id_errors = []
    for name, frame in [*inputs.items(), *((f"{s} labels", labels[s]) for s in ("val", "test"))]:
        if frame.columns.nlevels != 1 or not frame.columns.is_unique:
            errors.append(f"{name}: columns must be flat and unique.")
            continue
        if not all(isinstance(column, str) for column in frame.columns):
            errors.append(f"{name}: column names must be strings.")
        id_errors.extend(_id_errors(frame, name))
    errors.extend(id_errors)
    if any(frame.columns.nlevels != 1 or not frame.columns.is_unique for frame in inputs.values()):
        return errors
    if "y" not in inputs["train"]:
        errors.append("train: missing y column.")
    for split in ("val", "test"):
        if set(labels[split].columns) != {"id", "y"}:
            errors.append(f"{split} labels: expected exactly id and y columns.")
        if "y" in inputs[split]:
            errors.append(f"{split}: y must be stored only in the protected label table.")
    features = _feature_columns(inputs["train"])
    if not features:
        errors.append("train: at least one input feature is required.")
    for split, frame in inputs.items():
        if _feature_columns(frame) != features:
            errors.append(f"{split}: feature columns and order must match train.")
        for column in _feature_columns(frame):
            values = frame[column]
            if is_datetime64_any_dtype(values.dtype):
                valid = not values.isna().any()
            else:
                valid = _finite_numeric(values)
            if not valid:
                errors.append(
                    f"{split}: feature '{column}' must be finite numeric or non-null datetime data."
                )
            if column in features and split != "train":
                train_values = inputs["train"][column]
                if is_datetime64_any_dtype(values.dtype) != is_datetime64_any_dtype(train_values.dtype):
                    errors.append(f"{split}: feature '{column}' has an incompatible type with train.")
    ids_ready = not id_errors and all(
        "id" in frame and frame.columns.is_unique and frame.columns.nlevels == 1
        for frame in [*inputs.values(), *labels.values()]
    )
    if ids_ready:
        for split in ("val", "test"):
            if set(inputs[split]["id"]) != set(labels[split]["id"]):
                errors.append(
                    f"{split}: input and protected label IDs must match exactly; row order may differ."
                )
        for earlier, later in combinations(inputs, 2):
            if set(inputs[earlier]["id"]) & set(inputs[later]["id"]):
                errors.append(f"{earlier}/{later}: sample IDs overlap across splits.")
    if "feature_columns" in meta and meta["feature_columns"] != features:
        errors.append("meta.json: feature_columns contradicts the persisted train schema.")
    if "sizes" in meta and meta["sizes"] != {s: len(frame) for s, frame in inputs.items()}:
        errors.append("meta.json: sizes contradicts the persisted split sizes.")
    return errors


def _split_errors(inputs: dict, meta: dict, metadata_errors: list[str]) -> tuple[list, list, list]:
    """Check groups and per-entity row chronology; return errors for each check and notes."""
    strategy = meta.get("split")
    if metadata_errors or strategy not in ("group", "temporal"):
        errors = metadata_errors or ["meta.json: split must explicitly be 'group' or 'temporal'."]
        return errors, errors, []
    errors, notes = [], []
    if any(frame.columns.nlevels != 1 or not frame.columns.is_unique for frame in inputs.values()):
        return (
            ["Cannot verify groups with malformed columns."],
            ["Cannot verify times with malformed columns."],
            [],
        )
    group_col = meta.get("group_col")
    if group_col not in ("group", "unit"):
        errors.append(
            "meta.json: group_col must identify persisted 'group' or the verified raw 'unit' alias."
        )
    elif group_col == "unit":
        notes.append(
            "WARN: group_col='unit' names the raw source; checking persisted 'group', not assuming unit exists."
        )
    if "has_groups" in meta and meta["has_groups"] is not True:
        errors.append("meta.json: has_groups contradicts the groups required by this split strategy.")
    for split, frame in inputs.items():
        if "group" not in frame:
            errors.append(f"{split}: missing persisted group column.")
            continue
        groups = frame["group"]
        if groups.isna().any() or infer_dtype(groups, skipna=True) not in {"string", "integer", "empty"}:
            errors.append(f"{split}: groups must be non-null strings or integers.")
        if group_col == "unit" and "unit" in frame and not frame["unit"].eq(groups).all():
            errors.append(f"{split}: raw unit column contradicts persisted group.")
    if errors:
        if strategy == "group":
            return errors, [], [*notes, "NOT APPLICABLE: time_ordered; declared split is group-based."]
        return [], errors, [*notes, "NOT APPLICABLE: group_disjoint; temporal splits may reuse entities."]
    if strategy == "group":
        for earlier, later in combinations(inputs, 2):
            overlap = set(inputs[earlier]["group"]) & set(inputs[later]["group"])
            if overlap:
                errors.append(
                    f"{earlier}/{later}: {len(overlap)} group(s) overlap; split by entity before preparation."
                )
        return errors, [], [*notes, "NOT APPLICABLE: time_ordered; declared split is group-based."]

    notes.append("NOT APPLICABLE: group_disjoint; temporal splits may reuse entities.")
    notes.append(
        "WARN: only per-entity row times are checked; window boundaries and label horizons are not verified."
    )
    time_col = meta.get("time_col")
    if not isinstance(time_col, str) or time_col in {"id", "group", "y"}:
        return [], ["meta.json: temporal splits require a valid time_col."], notes
    ranges = {}
    for split, frame in inputs.items():
        if time_col not in frame:
            errors.append(f"{split}: missing temporal column '{time_col}'.")
            continue
        times = frame[time_col]
        if is_bool_dtype(times.dtype) or not (
            _finite_numeric(times) or is_datetime64_any_dtype(times.dtype) and not times.isna().any()
        ):
            errors.append(f"{split}: time_col must be finite numeric or non-null datetime data.")
            continue
        ranges[split] = frame.groupby("group")[time_col].agg(["min", "max"])
    if errors:
        return [], errors, notes
    kinds = {
        (
            is_datetime64_any_dtype(frame[time_col].dtype),
            getattr(frame[time_col].dtype, "tz", None) is not None,
        )
        for frame in inputs.values()
    }
    if len(kinds) != 1:
        return [], ["Temporal column types must be compatible across splits."], notes
    for earlier, later in combinations(inputs, 2):
        shared = ranges[earlier].index.intersection(ranges[later].index)
        if shared.empty:
            errors.append(
                f"{earlier}/{later}: no shared entity timeline; chronological separation cannot be verified."
            )
        elif not (ranges[earlier].loc[shared, "max"] < ranges[later].loc[shared, "min"]).all():
            errors.append(f"{earlier}/{later}: temporal ranges overlap or are reversed within an entity.")
    return [], errors, notes


def _size_errors(inputs: dict, ws: Workspace) -> list[str]:
    errors = [
        f"{split}: split must contain at least one sample." for split, frame in inputs.items() if frame.empty
    ]
    if ws.spec.prediction_format == "binary_score":
        errors.extend(
            f"{s}: two binary classes require at least two samples."
            for s, frame in inputs.items()
            if len(frame) < 2
        )
    metrics = {ws.spec.primary_metric, *ws.spec.metrics, *(k.metric for k in ws.spec.kpis)}
    if "r2" in metrics:
        errors.extend(
            f"{s}: r2 requires at least two evaluation samples."
            for s in ("val", "test")
            if len(inputs[s]) < 2
        )
    return errors


def _label_errors(labels: dict, ws: Workspace) -> list[str]:
    fmt = ws.spec.prediction_format
    if fmt not in {"scalar", "binary_score"}:
        return [f"Unsupported prediction format '{fmt}'; its ground-truth schema is not agreed yet."]
    errors = []
    metrics = {ws.spec.primary_metric, *ws.spec.metrics, *(k.metric for k in ws.spec.kpis)}
    for split, frame in labels.items():
        if "y" not in frame or not frame.columns.is_unique:
            errors.append(f"{split}: cannot validate labels without a unique y column.")
            continue
        targets = frame["y"]
        if not _finite_numeric(targets) or fmt == "scalar" and is_bool_dtype(targets.dtype):
            errors.append(f"{split}: targets must be finite, non-missing real numeric values.")
        elif fmt == "binary_score":
            if not targets.isin([0, 1]).all():
                errors.append(f"{split}: binary ground truth must be 0/1, not prediction scores.")
            elif targets.nunique() < 2:
                errors.append(f"{split}: binary training/evaluation requires both ground-truth classes.")
        elif split != "train" and "r2" in metrics and targets.nunique() < 2:
            errors.append(f"{split}: r2 requires non-constant evaluation targets.")
    return errors


def _leak_errors(inputs: dict, labels: dict, scan_ready: bool) -> list[str]:
    forbidden = {
        "y",
        "target",
        "label",
        "labels",
        "rul",
        "remaining_useful_life",
        "failure_cycle",
        "cycles_to_failure",
        "time_to_failure",
    }
    errors = []
    for split, frame in inputs.items():
        for column in frame.columns:
            if not isinstance(column, str) or column in {"id", "group"} or split == "train" and column == "y":
                continue
            name = column.rsplit(".", 1)[-1].lower()
            if name in forbidden or name.startswith(("target_", "label_", "y_")):
                errors.append(f"{split}: forbidden target/target-derived input column '{column}'.")
    if not scan_ready:
        errors.append("NOT VERIFIED: target-copy scan requires valid schemas, aligned IDs and valid targets.")
        return errors
    # shortcut: names and exact copies only; availability-time and indirect leakage need preparation metadata.
    for column in _feature_columns(inputs["train"]):
        if not all(is_numeric_dtype(frame[column].dtype) for frame in inputs.values()):
            continue
        copied = True
        for split, frame in inputs.items():
            aligned = labels[split].set_index("id")["y"].loc[frame["id"]]
            copied &= np.array_equal(frame[column].to_numpy(), aligned.to_numpy())
        if copied:
            errors.append(
                f"Input '{column}' exactly copies the ID-aligned target across all splits; inspect preparation."
            )
    return errors


def _data_card(ws: Workspace, inputs: dict, meta: dict, checks: dict, messages: list[str]) -> str:
    """Describe inputs and aggregate validation outcomes, without hidden label values."""
    lines = [
        f"# Dataset: {ws.spec.name}",
        f"- Task: {ws.task_id}",
        f"- Framing: {ws.spec.framing}",
        f"- Prediction format: {ws.spec.prediction_format}",
        f"- Requested strategy: {ws.spec.split_strategy}",
        f"- Persisted strategy: {meta.get('split') if isinstance(meta.get('split'), str) else 'unavailable'}",
        "",
        "## Inputs",
        "",
        "| Split | Samples | Groups | Missing feature values | Infinite feature values |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        frame = inputs.get(split)
        if frame is None or not frame.columns.is_unique or frame.columns.nlevels != 1:
            lines.append(f"| {split} | unavailable | unavailable | unavailable | unavailable |")
            continue
        features = frame[_feature_columns(frame)]
        missing = int(features.isna().sum().sum())
        infinite = sum(
            int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
            for _, values in features.items()
            if is_numeric_dtype(values.dtype) and not is_complex_dtype(values.dtype)
        )
        groups = "unavailable"
        if "group" in frame:
            groups = (
                frame["group"].nunique()
                if infer_dtype(frame["group"], skipna=True) in {"string", "integer", "empty"}
                else "invalid"
            )
        lines.append(f"| {split} | {len(frame)} | {groups} | {missing} | {infinite} |")
    train = inputs.get("train")
    if train is not None and train.columns.is_unique and train.columns.nlevels == 1:
        lines.extend(["", "## Feature schema", ""])
        lines.extend(f"- `{column}`: {train[column].dtype}" for column in _feature_columns(train))
    for key in ("group_col", "time_col"):
        value = meta.get(key)
        lines.append(f"- {key}: {value if isinstance(value, str) else 'unavailable'}")
    lines.extend(["", "## Validation", ""])
    for check, passed in checks.items():
        not_applicable = any(message.startswith(f"NOT APPLICABLE: {check};") for message in messages)
        status = "not applicable" if not_applicable else "passed" if passed else "failed / not verified"
        lines.append(f"- {check}: {status}")
    lines.extend(
        [
            "",
            *[f"- {message}" for message in messages],
            "",
            "## Limitations",
            "",
            "- Hidden per-example labels are omitted. No model performance is evaluated.",
            "- Leakage checks cover known names and exact target copies, not all statistical leakage.",
            "- Temporal checks compare row times within shared entities; windows/horizons are not certified.",
            "- IDs are checked as persisted; regenerated split IDs cannot prove original sample identity.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate(ws: Workspace) -> ValidationReport:
    """Validate all seven criteria without repairing data or persisting the report.

    Non-applicable strategy checks are True with explicit NOT APPLICABLE notes;
    required checks that cannot be verified are False. Expected file/data errors
    become diagnostics; unexpected programming errors propagate.
    """
    checks = dict.fromkeys(CHECKS, False)
    messages, inputs, meta = [], {}, {}
    data_dir = workspace.ws_dir(ws) / "data"
    labels_dir = workspace.labels_dir(ws.run_id, ws.task_id)
    flat = (data_dir / "train.parquet").exists()
    try:
        from pyarrow import ArrowException  # noqa: F401
    except ImportError:
        messages.append(
            "FAIL files_present: install the declared PyArrow dependency to read Parquet datasets."
        )
        loaded = {}
    else:
        if flat:
            loaded = _load_flat_layout(data_dir, labels_dir, messages)
        else:
            loaded = _load_harness_layout(data_dir, labels_dir, messages)
    paths = dict.fromkeys(("train", "val", "test", "val labels", "test labels"))
    if loaded and not flat and not _is_tabular(ws, data_dir):
        return _validate_generic(ws, loaded, data_dir, messages)
    inputs = {s: loaded[s] for s in ("train", "val", "test") if s in loaded}
    checks["files_present"] = len(loaded) == len(paths)
    metadata_errors = []
    try:
        meta = json.loads((data_dir / "meta.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        metadata_errors.append(
            f"meta.json is missing/unreadable ({type(exc).__name__}); write explicit split metadata."
        )
    if not isinstance(meta, dict):
        meta = {}
        metadata_errors.append("meta.json must contain a JSON object.")
    if meta.get("groups") and meta.get("group_col") not in ("group", "unit"):
        meta = {**meta, "group_col": "group"}  # groups come from write_dataset(groups=...), persisted as `group`
    if checks["files_present"]:
        labels = {"train": inputs["train"], "val": loaded["val labels"], "test": loaded["test labels"]}
        format_errors = _format_errors(inputs, labels, meta)
        label_errors = _label_errors(labels, ws)
        if ws.spec.prediction_format not in {"scalar", "binary_score"}:
            format_errors.append(
                "Unsupported prediction format; only scalar and binary_score schemas are validated."
            )
        group_errors, time_errors, notes = _split_errors(inputs, meta, metadata_errors)
        outcomes = {
            "format": format_errors,
            "group_disjoint": group_errors,
            "time_ordered": time_errors,
            "min_sizes": _size_errors(inputs, ws),
            "label_sanity": label_errors,
            "no_label_leak": _leak_errors(inputs, labels, not format_errors and not label_errors),
        }
        for check, errors in outcomes.items():
            checks[check] = not errors
            messages.extend(f"FAIL {check}: {error}" for error in errors)
        messages.extend(notes)
    else:
        messages.append("NOT VERIFIED: remaining checks require all five readable dataset tables.")
        messages.extend(metadata_errors)
    return ValidationReport(
        ok=all(checks.values()),
        checks=checks,
        messages=messages,
        data_card=_data_card(ws, inputs, meta, checks, messages),
    )


# ---------------------------------------------------------------- harness layout adapter
# harness.write_dataset (ML_PLAN.md) writes data/{split}_inputs.parquet + data/train_labels.parquet keyed by
# `sample_id`, hidden {labels_dir}/{split}_labels.parquet, and per-row groups in meta.json["groups"].
# The checks above work on frames with `id`, `group` and (train only) `y`; this builds those frames.

def _read_meta(data_dir) -> dict:
    try:
        meta = json.loads((data_dir / "meta.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _load_flat_layout(data_dir, labels_dir, messages: list[str]) -> dict:
    """Earlier layout (data/train.parquet with id / group / y; {labels_dir}/{split}.parquet); kept for its tests."""
    from pyarrow import ArrowException

    paths = {
        "train": data_dir / "train.parquet",
        "val": data_dir / "val_inputs.parquet",
        "test": data_dir / "test_inputs.parquet",
        "val labels": labels_dir / "val.parquet",
        "test labels": labels_dir / "test.parquet",
    }
    loaded = {}
    for name, path in paths.items():
        try:
            loaded[name] = pd.read_parquet(path, engine="pyarrow")
        except (OSError, ValueError, ImportError, ArrowException) as exc:
            messages.append(f"FAIL files_present: {name} table is missing/unreadable ({type(exc).__name__}).")
    return loaded


def _load_harness_layout(data_dir, labels_dir, messages: list[str]) -> dict:
    from pyarrow import ArrowException

    meta = _read_meta(data_dir)
    groups = meta.get("groups", {}) if isinstance(meta.get("groups"), dict) else {}
    loaded = {}

    def read(name, path):
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except (OSError, ValueError, ImportError, ArrowException) as exc:
            if path.with_suffix(".json").exists():  # boxes / masks labels
                return pd.DataFrame({"sample_id": list(json.loads(path.with_suffix(".json").read_text()))})
            messages.append(f"FAIL files_present: {name} table is missing/unreadable ({type(exc).__name__}).")
            return None
        return frame.rename(columns={"sample_id": "id"})

    for split in ("train", "val", "test"):
        frame = read(split, data_dir / f"{split}_inputs.parquet")
        if frame is None:
            continue
        g = groups.get(split)
        if g is not None and len(g) == len(frame) and "group" not in frame:
            frame.insert(1, "group", g)
        loaded[split] = frame
    train_y = read("train labels", data_dir / "train_labels.parquet")
    if "train" in loaded and train_y is not None:
        if "y" in train_y:
            loaded["train"] = loaded["train"].merge(train_y[["id", "y"]], on="id", how="left")
        else:
            loaded["train"]["y"] = np.nan
    elif train_y is None:
        loaded.pop("train", None)
    for split in ("val", "test"):
        frame = read(f"{split} labels", labels_dir / f"{split}_labels.parquet")
        if frame is not None:
            loaded[f"{split} labels"] = frame
    return loaded


def _is_tabular(ws: Workspace, data_dir) -> bool:
    kinds = _read_meta(data_dir).get("input_kind", {})
    return ws.spec.prediction_format in ("scalar", "binary_score") and all(
        k == "table" for k in (kinds.values() if isinstance(kinds, dict) else [])
    )


def _validate_generic(ws: Workspace, loaded: dict, data_dir, messages: list[str]) -> ValidationReport:
    """Image / signal / array inputs and class_probs / boxes / masks: the checks that apply to any layout.
    (The full tabular checks above cover scalar and binary_score on feature tables.)"""
    meta = _read_meta(data_dir)
    checks = dict.fromkeys(CHECKS, True)
    names = ("train", "val", "test", "val labels", "test labels")
    checks["files_present"] = all(n in loaded for n in names)
    if not checks["files_present"]:
        checks.update(dict.fromkeys(CHECKS, False))
        messages.append("NOT VERIFIED: remaining checks require all dataset tables.")
        return ValidationReport(ok=False, checks=checks, messages=messages)

    for split in ("val", "test"):
        if set(loaded[split]["id"]) != set(loaded[f"{split} labels"]["id"]):
            checks["format"] = False
            messages.append(f"FAIL format: {split} inputs and hidden labels must have the same sample_ids.")

    groups = {s: set(loaded[s]["group"]) for s in ("train", "val", "test") if "group" in loaded[s]}
    if meta.get("split") == "group" and len(groups) < 3:
        checks["group_disjoint"] = False
        messages.append("FAIL group_disjoint: split is 'group' but write_dataset got no groups for every split.")
    for a, b in combinations(groups, 2):
        overlap = groups[a] & groups[b]
        if overlap:
            checks["group_disjoint"] = False
            messages.append(f"FAIL group_disjoint: {a}/{b}: {len(overlap)} group(s) overlap; split by group.")

    sizes = {s: len(loaded[s]) for s in ("train", "val", "test")}
    for split, n in sizes.items():
        if n < (20 if split == "train" else 10):
            checks["min_sizes"] = False
            messages.append(f"FAIL min_sizes: {split} has only {n} samples.")

    fmt = ws.spec.prediction_format
    for split in ("val", "test"):
        labels = loaded[f"{split} labels"]
        if fmt in ("class_probs", "binary_score") and "y" in labels and labels["y"].nunique() < 2:
            checks["label_sanity"] = False
            messages.append(f"FAIL label_sanity: {split} labels contain a single class.")
    if fmt in ("boxes", "masks"):
        messages.append("NOT VERIFIED: label_sanity for boxes / masks (per-image annotations).")
    messages.append("NOT APPLICABLE: no_label_leak and time_ordered are checked for feature tables only.")

    kinds = meta.get("input_kind", {})
    card = "\n".join([
        f"# Data card: {ws.spec.name}",
        f"- Prediction format: {fmt}",
        f"- Input kind: {kinds}",
        f"- Sizes: {sizes}",
        f"- Classes: {meta.get('classes')}" if meta.get("classes") else "- Classes: n/a",
        f"- Split: {meta.get('split', 'unspecified')}; groups per split: { {s: len(g) for s, g in groups.items()} }",
        "",
        "## Checks",
        *(f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()),
    ])
    return ValidationReport(ok=all(checks.values()), checks=checks, messages=messages, data_card=card)
