"""Read-only validation of the persisted sensor datasets produced by prepare.py.

Supports legacy tables and separate-label tables/numeric arrays, including
class indices. Image files have integrity and exact pixel target-copy checks.
Split strategy comes from data/meta.json, never guessed from prose.
Hidden labels are used internally and are never included in diagnostics.
"""

import json
from itertools import combinations
from pathlib import Path

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
    "format",  # supported input/label schemas and sample identity match the persisted contract
    "group_disjoint",  # no group (engine unit, image id...) appears in more than one split
    "time_ordered",  # temporal splits: earlier max(time) < later min(time) within shared entities
    "min_sizes",  # each split above a minimum size
    "label_sanity",  # finite targets, valid class vocabulary and required class coverage
    "no_label_leak",  # known target names and exact ID-aligned target copies
]

TARGET_NAMES = {
    "y", "target", "label", "labels", "rul", "remaining_useful_life", "failure_cycle", "event_time",
    "cycles_to_failure", "time_to_failure",
}


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"id", "y"}
            and (column != "group" or frame.attrs.get("group_is_feature", False))]


def _finite_numeric(values: pd.Series) -> bool:
    """Real numeric values only; no string conversion or imputation."""
    return bool(
        is_numeric_dtype(values.dtype)
        and not is_complex_dtype(values.dtype)
        and np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan)).all()
    )


def _artifact_path(path: Path, root: Path) -> Path:
    """Reject redirects before opening dataset artifacts, including in-root symlinks."""
    try:
        contained = path.resolve().is_relative_to(root.resolve())
    except RuntimeError as exc:  # pathlib on Python 3.12 raises RuntimeError for symlink loops
        raise ValueError("dataset artifact has a symlink loop") from exc
    if (not contained
            or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))):
        raise ValueError("dataset artifact must remain under its directory without symlinks")
    return path


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
    for name, frame in [*inputs.items(), *((f"{s} labels", labels[s]) for s in labels)]:
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
    kinds = meta.get("input_kind", {})
    if not isinstance(kinds, dict):
        errors.append("meta.json: input_kind must be an object.")
        kinds = {}
    if not features and kinds.get("train") != "array":
        errors.append("train: at least one input feature is required.")
    for split, frame in inputs.items():
        if _feature_columns(frame) != features:
            errors.append(f"{split}: feature columns and order must match train.")
        for column in _feature_columns(frame):
            values = frame[column]
            if is_datetime64_any_dtype(values.dtype):
                valid = not values.isna().any()
            elif column == "path" and kinds.get(split) == "files":
                valid = not values.isna().any()  # containment and contents checked by _new_layout
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
        for split in labels:
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


def _random_split_errors(meta: dict) -> list[str]:
    """split='random' is only safe when every sample is an independent item (one photo per part). Tables,
    windows cut from a recording and frames from a video have near-duplicate neighbours, so they need groups."""
    kinds = meta.get("input_kind")
    kinds = set(kinds.values()) if isinstance(kinds, dict) else set()
    if kinds != {"files"}:
        return [("split='random' is only allowed for standalone image files; tables and arrays (windows, "
                 "sequences) need split='group' (by entity/recording) or 'temporal'.")]
    if meta.get("from_video"):
        return ["frames from a video are near-duplicates of their neighbours; use split='group' by clip."]
    if meta.get("groups"):
        return ["groups were given; declare split='group' so they are checked."]
    if meta.get("time_col"):
        return ["a time_col was given; use split='temporal'."]
    return []


def _random_split_ok(meta: dict) -> bool:
    return meta.get("split") == "random" and not _random_split_errors(meta)


def _duplicate_file_errors(inputs: dict, data_dir: Path) -> list[str]:
    """The same file content in two splits is a leak whatever the split strategy."""
    import hashlib

    seen = {}
    for split, frame in inputs.items():
        if "path" not in frame:
            return []
        for value in frame["path"]:
            try:
                digest = hashlib.sha256(_artifact_path(data_dir / value, data_dir).read_bytes()).hexdigest()
            except (OSError, ValueError, TypeError):
                continue  # unreadable files are reported by _file_errors
            seen.setdefault(digest, set()).add(split)
    dup = sum(len(s) > 1 for s in seen.values())
    return [f"{dup} file(s) have identical content in more than one split."] if dup else []


def _split_errors(inputs: dict, meta: dict, metadata_errors: list[str]) -> tuple[list, list, list]:
    """Check groups and per-entity row chronology; return errors for each check and notes."""
    strategy = meta.get("split")
    if not metadata_errors and strategy == "random":
        errors = _random_split_errors(meta)
        if errors:
            return errors, errors, []
        return [], [], [
            ("WARN: random split; samples are assumed independent (one file per physical item). "
             "Only exact duplicates across splits are checked."),
            "NOT APPLICABLE: time_ordered; declared split is random.",
        ]
    if metadata_errors or strategy not in ("group", "temporal"):
        errors = metadata_errors or [
            ("meta.json: split must explicitly be 'group' or 'temporal' ('random' only for standalone image files). "
            "In prepare.py pass it to the harness: harness.write_dataset(..., groups=..., meta={'split': 'group', ...}).")
        ]
        return errors, errors, []
    errors, notes = [], []
    if any(frame.columns.nlevels != 1 or not frame.columns.is_unique for frame in inputs.values()):
        return (
            ["Cannot verify groups with malformed columns."],
            ["Cannot verify times with malformed columns."],
            [],
        )
    group_col = meta.get("group_col")
    if "input_kind" in meta:
        if "group_col" in meta and (not isinstance(group_col, str) or not group_col.strip()):
            errors.append("meta.json: group_col must describe the source of persisted groups.")
    elif group_col not in ("group", "unit"):
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
        if "input_kind" in meta:
            stored = meta.get("groups", {})
            stored = stored.get(split) if isinstance(stored, dict) else None
            if (not isinstance(stored, list) or len(stored) != len(frame)
                    or not pd.Series(stored, index=frame.index).equals(groups)):
                errors.append(f"{split}: groups cannot be verified against row-aligned meta.json groups.")
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


def _label_errors(labels: dict, ws: Workspace, meta: dict | None = None) -> list[str]:
    fmt = ws.spec.prediction_format
    classes = (meta or {}).get("classes")
    if fmt not in {"scalar", "binary_score"} and not (fmt == "class_probs" and "input_kind" in (meta or {})):
        return [f"Unsupported prediction format '{fmt}'; ground-truth validation is not implemented."]
    errors = []
    if fmt == "class_probs" and (not isinstance(classes, list) or len(classes) < 2
                or any(not isinstance(c, (str, int, float)) or isinstance(c, bool) for c in classes)
                or len({json.dumps(c) for c in classes}) != len(classes)
                or any(isinstance(c, float) and not np.isfinite(c) for c in classes)):
        return ["meta.json: classes must be a unique, finite vocabulary of at least two classes."]
    if fmt == "class_probs" and meta.get("normal_class") is not None and meta["normal_class"] not in classes:
        errors.append("meta.json: normal_class must name a class in the persisted vocabulary.")
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
        elif fmt == "class_probs":
            if (is_bool_dtype(targets.dtype) or not targets.isin(range(len(classes))).all()
                    or not pd.api.types.is_integer_dtype(targets.dtype)):
                errors.append(f"{split}: targets must be integer indices into classes, not class names or scores.")
            elif (targets.nunique() < 2 or (split == "train" or "min_class_recall" in metrics)
                  and set(targets) != set(range(len(classes)))):
                errors.append(f"{split}: insufficient class coverage for training/requested evaluation metrics.")
        elif split != "train" and "r2" in metrics and targets.nunique() < 2:
            errors.append(f"{split}: r2 requires non-constant evaluation targets.")
    return errors


def _leak_errors(inputs: dict, labels: dict, scan_ready: bool, arrays: dict | None = None) -> list[str]:
    errors = []
    for split, frame in inputs.items():
        for column in frame.columns:
            if not isinstance(column, str) or column in {"id", "group"} or split == "train" and column == "y":
                continue
            name = column.rsplit(".", 1)[-1].lower()
            if name in TARGET_NAMES or name.startswith(("target_", "label_", "y_")):
                errors.append(f"{split}: forbidden target/target-derived input column '{column}'.")
    if not scan_ready:
        errors.append("NOT VERIFIED: target-copy scan requires valid schemas, non-empty inputs, aligned IDs and valid targets.")
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
    if arrays:
        copied = None
        for split, array in arrays.items():
            targets = labels[split].set_index("id")["y"].loc[inputs[split]["id"]].to_numpy()
            matches = (array.reshape(len(array), -1) == targets[:, None]).all(axis=0)
            copied = matches if copied is None else copied & matches
        if copied is not None and copied.any():
            errors.append("Array input exactly copies the ID-aligned target across all splits.")
    return errors


def _image_leak_errors(inputs: dict, labels: dict, data_dir: Path) -> list[str]:
    """Scan raw decoded pixels like array features, without retaining every image in memory."""
    from PIL import Image

    copied, shape = None, None
    for split, frame in inputs.items():
        targets = labels[split].set_index("id")["y"].loc[frame["id"]]
        for value, target in zip(frame["path"], targets, strict=True):
            try:
                with Image.open(_artifact_path(data_dir / value, data_dir)) as img:
                    for key in img.info:
                        name = key.rsplit(".", 1)[-1].lower()
                        if name in TARGET_NAMES or name.startswith(("target_", "label_", "y_")):
                            return [f"{split}: image metadata contains a forbidden target field."]
                    pixels = np.asarray(img)  # same raw decoding as catalog.load_image; also inspect metadata
            except (OSError, ValueError, ImportError, Image.DecompressionBombError) as exc:
                return [f"{split}: image target-copy scan NOT VERIFIED ({type(exc).__name__})."]
            if (pixels.ndim not in (2, 3) or not pixels.size or pixels.dtype.kind not in "biuf"
                    or not np.isfinite(pixels).all()):
                return [f"{split}: image target-copy scan requires finite real numeric pixels."]
            if shape is not None and pixels.shape != shape:
                return ["NOT VERIFIED: exact pixel target-copy scan requires consistent decoded image shapes."]
            shape = pixels.shape
            matches = pixels == target
            copied = matches if copied is None else copied & matches
    if copied is not None and copied.any():
        return ["Image pixel input exactly copies the ID-aligned target across all splits."]
    return []


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
    lines.extend(["", "## Split metadata (NOT input columns; groups come from harness.groups(split))", ""])
    for key in ("group_col", "time_col"):
        value = meta.get(key)
        lines.append(f"- {key}: {value if isinstance(value, str) else 'unavailable'}")
    if isinstance(meta.get("input_kind"), dict):
        for split in inputs:
            kind = meta["input_kind"].get(split)
            kind = kind if isinstance(kind, str) and kind in {"table", "array", "files"} else "unavailable"
            lines.append(f"- {split} input kind: {kind}")
        lines.append("- Table counts above describe identity/feature metadata; array contents are checked separately.")
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
            "- Image pixel scans use fixed decoded shapes; indirect leakage and embedded annotations are not certified.",
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
    messages, inputs, meta, arrays = [], {}, {}, {}
    data_dir = workspace.ws_dir(ws) / "data"
    labels_dir = workspace.labels_dir(ws.run_id, ws.task_id)
    new_layout = any((data_dir / name).exists() for name in (
        "train_inputs.parquet", "train_labels.parquet", "train_labels.json",
    ))
    paths = {
        "train": data_dir / ("train_inputs.parquet" if new_layout else "train.parquet"),
        "val": data_dir / "val_inputs.parquet",
        "test": data_dir / "test_inputs.parquet",
        "val labels": labels_dir / ("val_labels.parquet" if new_layout else "val.parquet"),
        "test labels": labels_dir / ("test_labels.parquet" if new_layout else "test.parquet"),
    }
    if new_layout:
        paths["train labels"] = data_dir / "train_labels.parquet"
    loaded = {}
    try:
        from pyarrow import ArrowException
    except ImportError:
        messages.append(
            "FAIL files_present: install the declared PyArrow dependency to read Parquet datasets."
        )
    else:
        for name, path in paths.items():
            try:
                root = labels_dir if name in {"val labels", "test labels"} else data_dir
                loaded[name] = pd.read_parquet(_artifact_path(path, root), engine="pyarrow")
            except (OSError, ValueError, ImportError, ArrowException) as exc:
                messages.append(
                    f"FAIL files_present: {name} table is missing/unreadable ({type(exc).__name__})."
                )
    inputs = {s: loaded[s] for s in ("train", "val", "test") if s in loaded}
    checks["files_present"] = len(loaded) == len(paths)
    metadata_errors = []
    try:
        meta = json.loads(_artifact_path(data_dir / "meta.json", data_dir).read_text())
    except (OSError, UnicodeError, ValueError) as exc:
        metadata_errors.append(
            f"meta.json is missing/unreadable ({type(exc).__name__}); write explicit split metadata."
        )
    if not isinstance(meta, dict):
        meta = {}
        metadata_errors.append("meta.json must contain a JSON object.")
    if checks["files_present"]:
        layout_errors = []
        if new_layout:
            inputs, labels, arrays, layout_errors, artifacts_ok = _new_layout(ws, loaded, meta)
            checks["files_present"] = artifacts_ok
            if not artifacts_ok:
                messages.append("FAIL files_present: dependent array/image artifacts are unreadable or not verified.")
            if (data_dir / "train.parquet").exists():
                layout_errors.append("Ambiguous dataset: legacy and separate-label train layouts coexist.")
            layout_errors.extend(metadata_errors)
        else:
            labels = {"train": inputs["train"], "val": loaded["val labels"], "test": loaded["test labels"]}
        format_errors = [*layout_errors, *_format_errors(inputs, labels, meta)]
        label_errors = _label_errors(labels, ws, meta)
        supported = {"scalar", "binary_score"} | ({"class_probs"} if new_layout else set())
        if ws.spec.prediction_format not in supported:
            format_errors.append(
                "Unsupported prediction format; scalar, binary_score and separate-label class_probs are validated."
            )
        group_errors, time_errors, notes = _split_errors(inputs, meta, metadata_errors)
        if new_layout and meta.get("input_kind", {}).get("train") == "files" and checks["files_present"]:
            group_errors = [*group_errors, *_duplicate_file_errors(inputs, data_dir)]
        scan_ready = not format_errors and not label_errors and all(len(f) for f in inputs.values())
        leak_errors = _leak_errors(inputs, labels, scan_ready, arrays)
        if scan_ready and any("path" in frame for frame in inputs.values()):
            if new_layout and meta["input_kind"].get("train") == "files":
                leak_errors.extend(_image_leak_errors(inputs, labels, data_dir))
                if not leak_errors:
                    notes.append("VERIFIED: neutral image paths and ID-aligned exact pixel target-copy scan.")
            else:
                leak_errors.append("NOT VERIFIED: file inputs require the verified separate-label image layout.")
        outcomes = {
            "format": format_errors,
            "group_disjoint": group_errors,
            "time_ordered": time_errors,
            "min_sizes": _size_errors(inputs, ws),
            "label_sanity": label_errors,
            "no_label_leak": leak_errors,
        }
        for check, errors in outcomes.items():
            checks[check] = not errors
            messages.extend(f"FAIL {check}: {error}" for error in errors)
        messages.extend(notes)
    else:
        messages.append("NOT VERIFIED: remaining checks require all readable dataset tables.")
        messages.extend(metadata_errors)
    if ws.spec.prediction_format in {"boxes", "masks"}:
        messages.append(f"Unsupported prediction format '{ws.spec.prediction_format}': structured labels are not validated.")
    for split, array in arrays.items():
        messages.append(f"INPUT {split}: array shape={array.shape}; dtype={array.dtype}; finite values verified.")
    if arrays:
        messages.append(
            "WARN: array row identity uses the writer's generated ordinal IDs; reordered array contents "
            "cannot be detected without writer-provided provenance/checksums."
        )
    return ValidationReport(
        ok=all(checks.values()),
        checks=checks,
        messages=messages,
        data_card=_data_card(ws, inputs, meta, checks, messages),
    )


def _new_layout(ws: Workspace, loaded: dict, meta: dict) -> tuple[dict, dict, dict, list[str], bool]:
    """Normalize separate-label tables in memory; array order is the companion table's row order."""
    inputs, labels, arrays, errors = {}, {}, {}, []
    artifacts_ok = True
    data_dir = workspace.ws_dir(ws) / "data"
    kinds = meta.get("input_kind")
    if not isinstance(kinds, dict) or set(kinds) != {"train", "val", "test"}:
        errors.append("meta.json: input_kind must declare table/files/array for every split.")
        kinds = {}
    if any(k not in {"table", "array", "files"} for k in kinds.values() if isinstance(k, str)) or any(
        not isinstance(k, str) for k in kinds.values()
    ):
        errors.append("meta.json: unsupported input_kind.")
    if len({k for k in kinds.values() if isinstance(k, str)}) > 1:
        errors.append("meta.json: input_kind must be consistent across splits.")
    meta["input_kind"] = kinds
    if meta.get("prediction_format") != ws.spec.prediction_format:
        errors.append("meta.json: prediction_format must match the frozen task.")
    if "sizes" not in meta:
        errors.append("meta.json: sizes must describe all persisted split sizes.")
    if "label_encoding" in meta:
        errors.append("meta.json: label_encoding is not defined by the current writer/scorer contract.")
    if any(f"{split}_labels" in meta for split in ("val", "test")):
        errors.append("Validation/test labels must not be embedded in public meta.json.")
    if any((data_dir / f"{s}{suffix}{ext}").exists()
           for s in ("val", "test") for suffix in ("", "_labels") for ext in (".parquet", ".json")):
        errors.append("Validation/test labels must not be persisted in the public data directory.")
    if any((data_dir / "files_y" / s).exists() for s in ("val", "test")):
        errors.append("Validation/test mask files must remain in the hidden labels directory.")
    groups = meta.get("groups")
    random_ok = groups is None and _random_split_ok(meta)  # no groups to align; _split_errors reports why
    if random_ok:
        groups = {}
    elif not isinstance(groups, dict) or set(groups) != {"train", "val", "test"}:
        errors.append("meta.json: groups must contain a row-aligned list for every split.")
        groups = {}
    for split in ("train", "val", "test"):
        for destination, name in ((inputs, split), (labels, f"{split} labels")):
            frame = loaded[name].copy()
            if "id" in frame or "sample_id" not in frame:
                errors.append(f"{name}: expected sample_id column; legacy id is not a feature.")
            destination[split] = frame.rename(columns={"sample_id": "id"})
            if destination is inputs:
                # The new loader exposes explicit group columns as features; injected metadata is not a feature.
                destination[split].attrs["group_is_feature"] = "group" in frame.columns
            if ("id" in destination[split] and destination[split].columns.is_unique
                    and infer_dtype(destination[split]["id"], skipna=True) not in {"string", "empty"}):
                errors.append(f"{name}: sample_id must be a string as generated by write_dataset.")
        frame = inputs[split]
        target = labels[split]
        if any(f.columns.nlevels != 1 or not f.columns.is_unique for f in (frame, target)):
            errors.append(f"{split}: columns must be flat and unique before metadata/label alignment.")
            continue
        if set(target.columns) != {"id", "y"}:
            errors.append(f"{split} labels: expected exactly sample_id and y columns.")
        if "y" in frame:
            errors.append(f"{split}: inputs must not contain y; labels belong in a separate table.")
        g = groups.get(split)
        if random_ok:
            pass
        elif (not isinstance(g, list) or len(g) != len(frame)
                or infer_dtype(g, skipna=False) not in {"string", "integer", "empty"}):
            errors.append(f"{split}: persisted groups must have one entry per input row.")
        elif frame.columns.is_unique:
            if "group" in frame and not frame["group"].eq(pd.Series(g, index=frame.index)).all():
                errors.append(f"{split}: input group contradicts row-aligned metadata groups.")
            frame["group"] = g
        if kinds.get(split) == "array":
            if set(frame.columns) - {"id", "group"}:
                errors.append(f"{split}: array companion table must contain only sample_id.")
            if ("id" in frame and frame.columns.is_unique
                    and frame["id"].tolist() != [f"{split}-{i:06d}" for i in range(len(frame))]):
                errors.append(f"{split}: array sample_id row order must match the writer's generated ordinal IDs.")
            path = data_dir / f"{split}_inputs.npy"
            try:
                arr = np.load(_artifact_path(path, data_dir), allow_pickle=False, mmap_mode="r")
                if (not isinstance(arr, np.ndarray) or arr.ndim < 1 or arr.shape[0] != len(frame)
                        or not all(arr.shape[1:]) or arr.dtype.kind not in "biuf"
                        or not np.isfinite(arr).all()):
                    raise ValueError("invalid numeric array")
                arrays[split] = arr
            except (OSError, ValueError, EOFError) as exc:
                artifacts_ok = False
                errors.append(f"{split}: array missing/unreadable or invalid ({type(exc).__name__}).")
        elif kinds.get(split) == "files":
            file_errors = _file_errors(frame, split, data_dir)
            artifacts_ok &= not file_errors
            errors.extend(file_errors)
        elif "path" in frame:
            errors.append(f"{split}: path inputs require input_kind='files'.")
        elif (data_dir / f"{split}_inputs.npy").exists():
            errors.append(f"{split}: unexpected array artifact contradicts input_kind.")
        if ("id" in frame and "id" in target and frame.columns.is_unique and target.columns.is_unique
                and not _id_errors(frame, split) and not _id_errors(target, f"{split} labels")
                and set(frame["id"]) == set(target["id"]) and "y" in target and split == "train"):
            frame["y"] = target.set_index("id")["y"].loc[frame["id"]].to_numpy()
    if arrays and (len(arrays) != 3 or len({a.shape[1:] for a in arrays.values()}) != 1):
        errors.append("Arrays must have matching per-sample shape across all splits.")
    return inputs, labels, arrays, errors, artifacts_ok



def _file_errors(frame: pd.DataFrame, split: str, data_dir: Path) -> list[str]:
    """Certify neutral local image files; recording decoders need a separate agreed schema."""
    if not frame.columns.is_unique or not {"id", "path"} <= set(frame.columns):
        return [f"{split}: files require unique sample_id and path columns."]
    try:
        from PIL import Image
    except ImportError:
        return [f"{split}: image contents NOT VERIFIED; install the declared Pillow dependency."]
    errors = []
    for sid, value in zip(frame["id"], frame["path"], strict=True):
        if not isinstance(sid, str) or not isinstance(value, str):
            errors.append(f"{split}: file identity/path must be non-null strings.")
            continue
        path = Path(value)
        expected = Path("files") / split / f"{sid}{path.suffix}"
        if Path(sid).name != sid or path.is_absolute() or path != expected:
            errors.append(f"{split}: input paths must be neutral relative files/<split>/<sample_id>.<ext>.")
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            errors.append(f"{split}: recording/file contents are NOT VERIFIED; only image files are validated.")
            continue
        try:
            local = _artifact_path(data_dir / path, data_dir)
            with Image.open(local) as img:
                img.verify()
            with Image.open(local) as img:
                img.load()
        except (OSError, ValueError, ImportError, Image.DecompressionBombError) as exc:
            errors.append(f"{split}: image missing/unreadable ({type(exc).__name__}).")
    return list(dict.fromkeys(errors))
