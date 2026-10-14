"""Real validator tests against synthetic persisted Parquet tables."""

import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import settings
from core.contracts import TaskSpec, ValidationReport, Workspace
from ml import validators, workspace


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    spec = TaskSpec(
        id="sensor_task",
        name="Sensor regression",
        rationale="Prevent downtime",
        framing="Estimate degradation",
        input_streams=["temperature"],
        label_sources=["maintenance_log"],
        target_definition="Prepared regression target",
        prediction_format="scalar",
        primary_metric="mae",
        metrics=["mae"],
        kpis=[],
        split_strategy="by engine",
        time_budget_s=60,
    )
    ws = Workspace(run_id="run", task_id=spec.id, path=str(settings.runs_dir / "run" / spec.id), spec=spec)
    inputs, labels = {}, {}
    for group, split in enumerate(("train", "val", "test"), 1):
        ids = [f"{split}-{i:06d}" for i in range(4)]
        inputs[split] = pd.DataFrame(
            {"id": ids, "group": [group] * 4, "cycle": [1, 2, 3, 4], "temperature": [11.0, 12.0, 13.0, 14.0]}
        )
        labels[split] = pd.DataFrame({"id": ids, "y": [-3.0, -1.0, 1.0, 3.0]})
    meta = {
        "split": "group",
        "group_col": "group",
        "time_col": "cycle",
        "has_groups": True,
        "sizes": dict.fromkeys(inputs, 4),
        "feature_columns": ["cycle", "temperature"],
    }
    return {"ws": ws, "inputs": inputs, "labels": labels, "meta": meta}


def _persist(data):
    ws = data["ws"]
    folder = workspace.ws_dir(ws) / "data"
    protected = workspace.labels_dir(ws.run_id, ws.task_id)
    folder.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    # Like Samu's writer, train y belongs to its corresponding input rows.
    train = data["inputs"]["train"].assign(y=data["labels"]["train"]["y"].to_numpy())
    train.to_parquet(folder / "train.parquet", index=False)
    for split in ("val", "test"):
        data["inputs"][split].to_parquet(folder / f"{split}_inputs.parquet", index=False)
        data["labels"][split].to_parquet(protected / f"{split}.parquet", index=False)
    (folder / "meta.json").write_text(json.dumps(data["meta"]))


def _report(data):
    _persist(data)
    return validators.validate(data["ws"])


def _temporal(data):
    data["meta"].update(split="temporal")
    data["ws"] = data["ws"].model_copy(
        update={
            "spec": data["ws"].spec.model_copy(update={"split_strategy": "by time within each engine"}),
        }
    )
    for phase, split in enumerate(("train", "val", "test")):
        data["inputs"][split]["group"] = [1, 1, 2, 2]
        # Engine 2 has a different cycle domain: global ordering would incorrectly fail.
        data["inputs"][split]["cycle"] = [1 + phase * 2, 2 + phase * 2, 101 + phase * 2, 102 + phase * 2]


def _binary(data):
    spec = data["ws"].spec.model_copy(
        update={
            "prediction_format": "binary_score",
            "primary_metric": "auroc",
            "metrics": ["auroc"],
        }
    )
    data["ws"] = data["ws"].model_copy(update={"spec": spec})
    for frame in data["labels"].values():
        frame["y"] = [0, 1, 0, 1]


def test_valid_group_split_and_pydantic_contract(dataset):
    report = _report(dataset)
    assert isinstance(report, ValidationReport)
    assert report.ok and report.checks == dict.fromkeys(validators.CHECKS, True)
    assert "NOT APPLICABLE: time_ordered" in "\n".join(report.messages)
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report
    assert set(report.model_dump()) == {"ok", "checks", "messages", "data_card", "log_tail"}
    assert report.log_tail == ""


@pytest.mark.parametrize(
    "location, filename",
    [
        ("data", "train.parquet"),
        ("data", "val_inputs.parquet"),
        ("data", "test_inputs.parquet"),
        ("labels", "val.parquet"),
        ("labels", "test.parquet"),
    ],
)
def test_missing_required_files(dataset, location, filename):
    _persist(dataset)
    ws = dataset["ws"]
    folder = (
        workspace.ws_dir(ws) / "data" if location == "data" else workspace.labels_dir(ws.run_id, ws.task_id)
    )
    (folder / filename).unlink()
    report = validators.validate(ws)
    assert not report.ok and not report.checks["files_present"]
    assert "missing/unreadable" in "\n".join(report.messages)
    assert "NOT VERIFIED" in "\n".join(report.messages)


@pytest.mark.parametrize("name", ["train.parquet", "val_inputs.parquet"])
def test_corrupt_parquet_is_reported_without_disclosing_contents(dataset, name):
    _persist(dataset)
    (workspace.ws_dir(dataset["ws"]) / "data" / name).write_bytes(b"SECRET_PROTECTED_PAYLOAD_not_parquet")
    report = validators.validate(dataset["ws"])
    assert not report.ok and not report.checks["files_present"]
    assert "SECRET_PROTECTED_PAYLOAD" not in report.model_dump_json()


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("problem", ["missing", "null", "duplicate", "blank", "floating", "nested"])
def test_invalid_input_ids(dataset, split, problem):
    frame = dataset["inputs"][split]
    if problem == "missing":
        dataset["inputs"][split] = frame.drop(columns="id")
    elif problem == "null":
        frame.loc[0, "id"] = None
    elif problem == "duplicate":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    elif problem == "blank":
        frame.loc[0, "id"] = " "
    elif problem == "floating":
        frame["id"] = [1.1, 2.2, 3.3, 4.4]
    else:
        frame["id"] = [[1], [2], [3], [4]]
    report = _report(dataset)
    assert not report.ok and not report.checks["format"]
    assert "id" in "\n".join(report.messages).lower()


@pytest.mark.parametrize("problem", ["missing", "null", "duplicate", "mismatch"])
def test_invalid_protected_label_ids(dataset, problem):
    frame = dataset["labels"]["val"]
    if problem == "missing":
        dataset["labels"]["val"] = frame.drop(columns="id")
    elif problem == "null":
        frame.loc[0, "id"] = None
    elif problem == "duplicate":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    else:
        frame.loc[0, "id"] = "other-sample"
    report = _report(dataset)
    assert not report.ok and not report.checks["format"]


def test_sample_ids_must_not_overlap_across_splits(dataset):
    dataset["inputs"]["val"]["id"] = dataset["inputs"]["train"]["id"].to_numpy()
    dataset["labels"]["val"]["id"] = dataset["inputs"]["val"]["id"].to_numpy()
    report = _report(dataset)
    assert not report.ok and not report.checks["format"]
    assert "sample IDs overlap" in "\n".join(report.messages)


def test_label_row_order_does_not_matter(dataset):
    dataset["labels"]["val"] = dataset["labels"]["val"].iloc[::-1].reset_index(drop=True)
    dataset["labels"]["test"] = (
        dataset["labels"]["test"].sample(frac=1, random_state=7).reset_index(drop=True)
    )
    assert _report(dataset).ok


@pytest.mark.parametrize("earlier, later", [("train", "val"), ("val", "test"), ("train", "test")])
def test_group_overlap(dataset, earlier, later):
    dataset["inputs"][later]["group"] = dataset["inputs"][earlier]["group"].to_numpy()
    report = _report(dataset)
    assert not report.ok and not report.checks["group_disjoint"]
    assert f"{earlier}/{later}" in "\n".join(report.messages)


def test_verified_raw_unit_alias_uses_persisted_group(dataset):
    dataset["meta"]["group_col"] = "unit"
    report = _report(dataset)
    assert report.ok and report.checks["group_disjoint"]
    assert "raw source" in "\n".join(report.messages)


def test_raw_unit_alias_does_not_hide_group_leakage(dataset):
    dataset["meta"]["group_col"] = "unit"
    dataset["inputs"]["val"]["group"] = dataset["inputs"]["train"]["group"].to_numpy()
    assert not _report(dataset).checks["group_disjoint"]


def test_contradictory_raw_unit_column_is_rejected(dataset):
    dataset["meta"]["group_col"] = "unit"
    dataset["meta"]["feature_columns"].append("unit")
    for frame in dataset["inputs"].values():
        frame["unit"] = 99
    report = _report(dataset)
    assert not report.ok and not report.checks["group_disjoint"]
    assert "contradicts persisted group" in "\n".join(report.messages)


@pytest.mark.parametrize(
    "problem", ["missing", "null", "nested", "metadata_missing", "metadata_unknown", "flag"]
)
def test_missing_or_invalid_groups(dataset, problem):
    if problem == "missing":
        dataset["inputs"]["val"] = dataset["inputs"]["val"].drop(columns="group")
    elif problem == "null":
        dataset["inputs"]["val"]["group"] = [None, 2, 2, 2]
    elif problem == "nested":
        dataset["inputs"]["val"]["group"] = [[2]] * 4
    elif problem == "metadata_missing":
        dataset["meta"].pop("group_col")
    elif problem == "metadata_unknown":
        dataset["meta"]["group_col"] = "engine"
    else:
        dataset["meta"]["has_groups"] = False
    report = _report(dataset)
    assert not report.ok and not report.checks["group_disjoint"]


def test_valid_temporal_split_reuses_entities_and_compares_cycles_per_engine(dataset):
    _temporal(dataset)
    report = _report(dataset)
    assert report.ok and report.checks["time_ordered"]
    assert "NOT APPLICABLE: group_disjoint" in "\n".join(report.messages)
    assert "window boundaries and label horizons are not verified" in "\n".join(report.messages)


@pytest.mark.parametrize("earlier, later", [("train", "val"), ("val", "test"), ("train", "test")])
@pytest.mark.parametrize("relation", ["equal", "reversed"])
def test_temporal_overlap_or_reversed_order(dataset, earlier, later, relation):
    _temporal(dataset)
    offset = 0 if relation == "equal" else -10
    dataset["inputs"][later]["cycle"] = dataset["inputs"][earlier]["cycle"] + offset
    report = _report(dataset)
    assert not report.ok and not report.checks["time_ordered"]


@pytest.mark.parametrize(
    "problem", ["missing_metadata", "missing_column", "nonfinite", "missing_group", "no_domain"]
)
def test_temporal_checks_require_interpretable_metadata_and_entity_domain(dataset, problem):
    _temporal(dataset)
    if problem == "missing_metadata":
        dataset["meta"].pop("time_col")
    elif problem == "missing_column":
        dataset["meta"]["time_col"] = "timestamp"
    elif problem == "nonfinite":
        dataset["inputs"]["val"]["cycle"] = [np.inf, 4, 103, 104]
    elif problem == "missing_group":
        dataset["inputs"]["val"] = dataset["inputs"]["val"].drop(columns="group")
    else:
        dataset["inputs"]["val"]["group"] = [3, 3, 4, 4]
    report = _report(dataset)
    assert not report.ok and not report.checks["time_ordered"]


def test_datetime_temporal_column(dataset):
    _temporal(dataset)
    for frame in dataset["inputs"].values():
        frame["cycle"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(frame["cycle"], unit="h")
    assert _report(dataset).ok


def test_naive_and_aware_time_domains_are_rejected_with_report(dataset):
    _temporal(dataset)
    for split, frame in dataset["inputs"].items():
        start = pd.Timestamp("2026-01-01", tz="UTC" if split != "val" else None)
        frame["cycle"] = start + pd.to_timedelta(frame["cycle"], unit="h")
    report = _report(dataset)
    assert not report.ok and not report.checks["time_ordered"]
    assert "Temporal column types" in "\n".join(report.messages)


@pytest.mark.parametrize("metadata", [{}, [], {"split": "random"}, {"split": None}])
def test_missing_or_unknown_split_strategy_does_not_block_independent_checks(dataset, metadata):
    dataset["meta"] = metadata
    report = _report(dataset)
    assert not report.ok
    assert not report.checks["group_disjoint"] and not report.checks["time_ordered"]
    assert report.checks["format"] and report.checks["label_sanity"] and report.checks["no_label_leak"]


@pytest.mark.parametrize("problem", ["missing", "corrupt"])
def test_unreadable_metadata_is_reported(dataset, problem):
    _persist(dataset)
    path = workspace.ws_dir(dataset["ws"]) / "data" / "meta.json"
    if problem == "missing":
        path.unlink()
    else:
        path.write_text("{bad json")
    report = validators.validate(dataset["ws"])
    assert not report.ok and report.checks["files_present"] and report.checks["format"]
    assert not report.checks["group_disjoint"] and "meta.json" in "\n".join(report.messages)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_empty_splits(dataset, split):
    dataset["inputs"][split] = dataset["inputs"][split].iloc[:0]
    dataset["labels"][split] = dataset["labels"][split].iloc[:0]
    dataset["meta"]["sizes"][split] = 0
    report = _report(dataset)
    assert not report.ok and not report.checks["min_sizes"]


def test_one_regression_evaluation_sample_is_valid_for_mae(dataset):
    dataset["inputs"]["val"] = dataset["inputs"]["val"].iloc[:1]
    dataset["labels"]["val"] = dataset["labels"]["val"].iloc[:1]
    dataset["meta"]["sizes"]["val"] = 1
    assert _report(dataset).ok


def test_r2_requires_two_evaluation_samples(dataset):
    dataset["ws"] = dataset["ws"].model_copy(
        update={
            "spec": dataset["ws"].spec.model_copy(update={"primary_metric": "r2", "metrics": ["r2"]}),
        }
    )
    dataset["inputs"]["val"] = dataset["inputs"]["val"].iloc[:1]
    dataset["labels"]["val"] = dataset["labels"]["val"].iloc[:1]
    dataset["meta"]["sizes"]["val"] = 1
    report = _report(dataset)
    assert not report.ok and not report.checks["min_sizes"]


def test_r2_requires_nonconstant_evaluation_targets(dataset):
    dataset["ws"] = dataset["ws"].model_copy(
        update={
            "spec": dataset["ws"].spec.model_copy(update={"metrics": ["mae", "r2"]}),
        }
    )
    dataset["labels"]["test"]["y"] = 0.0
    assert not _report(dataset).checks["label_sanity"]


@pytest.mark.parametrize("target", [np.nan, np.inf, -np.inf, "bad", [1, 2], True])
def test_invalid_scalar_targets(dataset, target):
    dataset["labels"]["val"]["y"] = [target] * 4
    report = _report(dataset)
    assert not report.ok and not report.checks["label_sanity"]


def test_complex_target_is_rejected_by_real_label_check(dataset):
    # Complex values cannot be written to Parquet, so exercise the label checker directly.
    dataset["labels"]["val"]["y"] = [1 + 2j] * 4
    assert validators._label_errors(dataset["labels"], dataset["ws"])


def test_negative_and_constant_regression_targets_are_allowed(dataset):
    for frame in dataset["labels"].values():
        frame["y"] = -5.0
    assert _report(dataset).ok


def test_valid_binary_numeric_ground_truth(dataset):
    _binary(dataset)
    assert _report(dataset).ok


def test_valid_binary_boolean_ground_truth(dataset):
    _binary(dataset)
    for frame in dataset["labels"].values():
        frame["y"] = [False, True, False, True]
    assert _report(dataset).ok


@pytest.mark.parametrize(
    "values", [[0, 2, 0, 1], [-1, 1, 0, 1], [0.1, 0.8, 0.2, 0.9], ["0", "1", "0", "1"], [0, np.nan, 0, 1]]
)
def test_invalid_binary_targets(dataset, values):
    _binary(dataset)
    dataset["labels"]["val"]["y"] = values
    assert not _report(dataset).checks["label_sanity"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_binary_single_class_is_rejected(dataset, split):
    _binary(dataset)
    dataset["labels"][split]["y"] = 0
    report = _report(dataset)
    assert not report.ok and not report.checks["label_sanity"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_missing_target_column(dataset, split):
    _persist(dataset)
    ws = dataset["ws"]
    path = (
        workspace.ws_dir(ws) / "data" / "train.parquet"
        if split == "train"
        else workspace.labels_dir(ws.run_id, ws.task_id) / f"{split}.parquet"
    )
    frame = pd.read_parquet(path).drop(columns="y")
    frame.to_parquet(path, index=False)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["label_sanity"]


@pytest.mark.parametrize(
    "column", ["y", "target", "rul", "failure_cycle", "cycles_to_failure", "label_future"]
)
def test_explicit_target_leakage(dataset, column):
    dataset["inputs"]["val"][column] = [0, 1, 2, 3]
    report = _report(dataset)
    assert not report.ok and not report.checks["no_label_leak"]
    assert column in "\n".join(report.messages)


def test_anonymous_target_copy_is_detected_by_id_with_shuffled_labels(dataset):
    dataset["meta"]["feature_columns"].append("proxy")
    for split, frame in dataset["inputs"].items():
        frame["proxy"] = dataset["labels"][split].set_index("id")["y"].loc[frame["id"]].to_numpy()
    dataset["labels"]["val"] = dataset["labels"]["val"].iloc[::-1]
    report = _report(dataset)
    assert report.checks["format"] and not report.checks["no_label_leak"]
    assert "ID-aligned target" in "\n".join(report.messages)


@pytest.mark.parametrize(
    "problem", ["missing", "extra", "order", "string", "nested", "datetime_type", "no_features"]
)
def test_invalid_feature_schemas(dataset, problem):
    frame = dataset["inputs"]["val"]
    if problem == "missing":
        dataset["inputs"]["val"] = frame.drop(columns="temperature")
    elif problem == "extra":
        frame["extra"] = 1.0
    elif problem == "order":
        dataset["inputs"]["val"] = frame[["id", "group", "temperature", "cycle"]]
    elif problem == "string":
        frame["temperature"] = ["a", "b", "c", "d"]
    elif problem == "nested":
        frame["temperature"] = [[1], [2], [3], [4]]
    elif problem == "datetime_type":
        frame["temperature"] = pd.date_range("2026-01-01", periods=4)
    else:
        for split in dataset["inputs"]:
            dataset["inputs"][split] = dataset["inputs"][split][["id", "group"]]
        dataset["meta"]["feature_columns"] = []
    assert not _report(dataset).checks["format"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_require_preparation_fix(dataset, value):
    dataset["inputs"]["val"]["temperature"] = [value, 12, 13, 14]
    report = _report(dataset)
    assert not report.ok and not report.checks["format"]
    assert "finite numeric" in "\n".join(report.messages)


@pytest.mark.parametrize("field, value", [("sizes", {"train": 99}), ("feature_columns", ["wrong"])])
def test_metadata_contradictions_are_reported(dataset, field, value):
    dataset["meta"][field] = value
    assert not _report(dataset).checks["format"]


@pytest.mark.parametrize("fmt", ["class_probs", "boxes", "masks"])
def test_unsupported_prediction_formats_fail_explicitly(dataset, fmt):
    dataset["ws"] = dataset["ws"].model_copy(
        update={
            "spec": dataset["ws"].spec.model_copy(update={"prediction_format": fmt}),
        }
    )
    report = _report(dataset)
    assert not report.ok and not report.checks["format"] and not report.checks["label_sanity"]
    assert f"Unsupported prediction format '{fmt}'" in "\n".join(report.messages)


def test_data_card_contains_markdown_schema_counts_results_and_limits(dataset):
    report = _report(dataset)
    card = report.data_card
    assert card.startswith("# Dataset: Sensor regression")
    assert "scalar" in card and "temperature" in card
    assert "| train | 4 | 1 | 0 | 0 |" in card
    assert "group_col: group" in card and "time_col: cycle" in card
    assert "time_ordered: not applicable" in card and "format: passed" in card
    assert "## Limitations" in card
    assert not (workspace.ws_dir(dataset["ws"]) / "data_card.md").exists()


def test_hidden_per_example_labels_and_arbitrary_metadata_are_never_reported(dataset):
    secrets = [91427.123456, 82538.234567, 73649.345678, 64750.456789]
    for split in ("val", "test"):
        dataset["labels"][split]["y"] = secrets
    dataset["meta"]["extra_private_metadata"] = "PRIVATE_EVALUATION_ANSWER"
    report = _report(dataset)
    assert report.ok
    text = report.model_dump_json()
    assert "PRIVATE_EVALUATION_ANSWER" not in text
    assert all(str(value) not in text for value in secrets)


def test_validation_never_modifies_files_or_creates_artifacts_and_is_deterministic(dataset):
    _persist(dataset)
    root = settings.runs_dir

    def snapshot():
        return {
            str(path.relative_to(root)): (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    first = validators.validate(dataset["ws"])
    assert validators.validate(dataset["ws"]) == first
    assert snapshot() == before


def test_expected_permission_error_produces_report_without_exception_payload(dataset, monkeypatch):
    _persist(dataset)

    def denied(*args, **kwargs):
        raise PermissionError("PRIVATE_LABEL_VALUE")

    monkeypatch.setattr(pd, "read_parquet", denied)
    report = validators.validate(dataset["ws"])
    assert not report.ok and not report.checks["files_present"]
    assert (
        "PermissionError" in report.model_dump_json()
        and "PRIVATE_LABEL_VALUE" not in report.model_dump_json()
    )


def test_missing_parquet_engine_is_an_actionable_failure(dataset, monkeypatch):
    _persist(dataset)
    real_import = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyarrow" and "ArrowException" in fromlist:
            raise ImportError("engine unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    report = validators.validate(dataset["ws"])
    assert not report.ok and not report.checks["files_present"]
    assert "install the declared PyArrow" in "\n".join(report.messages)


@pytest.mark.parametrize("table", ["train.parquet", "val.parquet"])
def test_malformed_duplicate_columns_do_not_crash_validation(dataset, monkeypatch, table):
    _persist(dataset)
    real_read = pd.read_parquet

    def read(path, **kwargs):
        frame = real_read(path, **kwargs)
        if path.name == table:
            return pd.concat([frame, frame[["id"]]], axis=1)
        return frame

    monkeypatch.setattr(pd, "read_parquet", read)
    report = validators.validate(dataset["ws"])
    assert not report.ok and not report.checks["format"]
    assert "flat and unique" in "\n".join(report.messages)


def test_invalid_hidden_label_values_are_not_in_failure_messages(dataset):
    dataset["labels"]["val"]["y"] = "PRIVATE_INVALID_GROUND_TRUTH"
    report = _report(dataset)
    assert not report.ok and "PRIVATE_INVALID_GROUND_TRUTH" not in report.model_dump_json()


def test_unexpected_programming_errors_are_not_hidden(dataset, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(pd, "read_parquet", broken)
    with pytest.raises(RuntimeError, match="programming bug"):
        validators.validate(dataset["ws"])


@pytest.fixture
def separate_dataset(dataset):
    data = workspace.ws_dir(dataset["ws"]) / "data"
    hidden = workspace.labels_dir(dataset["ws"].run_id, dataset["ws"].task_id)
    data.mkdir(parents=True)
    hidden.mkdir(parents=True)
    meta = {"prediction_format": "scalar", "input_kind": dict.fromkeys(dataset["inputs"], "table"),
            "sizes": dict.fromkeys(dataset["inputs"], 4), "split": "group", "group_col": "entity",
            "time_col": "cycle", "groups": {s: f["group"].tolist() for s, f in dataset["inputs"].items()}}
    for split in dataset["inputs"]:
        dataset["inputs"][split].drop(columns="group").rename(columns={"id": "sample_id"}).to_parquet(
            data / f"{split}_inputs.parquet", index=False)
        dataset["labels"][split].rename(columns={"id": "sample_id"}).to_parquet(
            (data if split == "train" else hidden) / f"{split}_labels.parquet", index=False)
    (data / "meta.json").write_text(json.dumps(meta))
    return dataset["ws"], data, hidden, meta


def _write_meta(data, meta):
    (data / "meta.json").write_text(json.dumps(meta))


def _arrays(data, meta):
    for split in ("train", "val", "test"):
        f = pd.read_parquet(data / f"{split}_inputs.parquet")
        f[["sample_id"]].to_parquet(data / f"{split}_inputs.parquet", index=False)
        np.save(data / f"{split}_inputs.npy", np.arange(24, dtype="float32").reshape(4, 2, 3))
    meta["input_kind"] = dict.fromkeys(("train", "val", "test"), "array")
    _write_meta(data, meta)


def test_separate_labels_metadata_groups_and_train_alignment(separate_dataset):
    ws, data, _hidden, meta = separate_dataset
    labels = pd.read_parquet(data / "train_labels.parquet")
    labels.iloc[::-1].to_parquet(data / "train_labels.parquet", index=False)
    report = validators.validate(ws)
    assert report.ok, report.messages
    assert "| train | 4 | 1 | 0 | 0 |" in report.data_card
    meta["groups"]["test"] = meta["groups"]["train"]
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["group_disjoint"]


@pytest.mark.parametrize("problem", ["kind_missing", "kind_invalid", "kind_mixed", "format", "groups_missing",
                                      "groups_short", "groups_nested", "sizes", "encoding", "train_ids",
                                      "public_labels", "ambiguous", "input_y", "index_only", "duplicate_id"])
def test_new_contract_corruption_fails_closed(separate_dataset, problem):
    ws, data, hidden, meta = separate_dataset
    if problem == "kind_missing":
        meta.pop("input_kind")
    elif problem == "kind_invalid":
        meta["input_kind"]["val"] = ["table"]
    elif problem == "kind_mixed":
        meta["input_kind"]["val"] = "array"
    elif problem == "format":
        meta["prediction_format"] = "binary_score"
    elif problem == "groups_missing":
        meta.pop("groups")
    elif problem == "groups_short":
        meta["groups"]["val"] = [2]
    elif problem == "groups_nested":
        meta["groups"]["val"] = [[2]] * 4
    elif problem == "sizes":
        meta.pop("sizes")
    elif problem == "encoding":
        meta["label_encoding"] = "unknown"
    elif problem == "train_ids":
        f = pd.read_parquet(data / "train_labels.parquet")
        f.loc[0, "sample_id"] = "wrong"
        f.to_parquet(data / "train_labels.parquet", index=False)
    elif problem == "public_labels":
        (data / "val_labels.parquet").write_bytes((hidden / "val_labels.parquet").read_bytes())
    elif problem == "ambiguous":
        (data / "train.parquet").write_bytes((data / "train_inputs.parquet").read_bytes())
    else:
        f = pd.read_parquet(data / "val_inputs.parquet")
        if problem == "input_y":
            f["y"] = 0
        elif problem == "index_only":
            f = f.set_index("sample_id")
        else:
            f.loc[1, "sample_id"] = f.loc[0, "sample_id"]
        f.to_parquet(data / "val_inputs.parquet", index=problem == "index_only")
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok, report.messages
    assert not report.checks["no_label_leak"]


def test_new_temporal_split_compares_independent_entity_times(separate_dataset):
    ws, data, _, meta = separate_dataset
    meta.update(split="temporal", time_col="cycle")
    for phase, split in enumerate(("train", "val", "test")):
        f = pd.read_parquet(data / f"{split}_inputs.parquet")
        f["cycle"] = [1 + phase * 2, 2 + phase * 2, 101 + phase * 2, 102 + phase * 2]
        f.to_parquet(data / f"{split}_inputs.parquet", index=False)
        meta["groups"][split] = [1, 1, 2, 2]
    _write_meta(data, meta)
    assert validators.validate(ws).ok
    f["cycle"] = [1, 2, 101, 102]
    f.to_parquet(data / "test_inputs.parquet", index=False)
    assert not validators.validate(ws).checks["time_ordered"]


def test_array_shape_dtype_finiteness_and_summary(separate_dataset):
    ws, data, _, meta = separate_dataset
    _arrays(data, meta)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}
    report = validators.validate(ws)
    assert report.ok, report.messages
    assert "shape=(4, 2, 3); dtype=float32" in report.data_card
    assert "reordered array contents" in report.data_card
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}


@pytest.mark.parametrize("problem", ["missing", "corrupt", "short", "shape", "dtype", "nan", "object",
                                      "zero_width", "scalar", "companion_feature", "temporal_without_time"])
def test_invalid_array_integrity(separate_dataset, problem):
    ws, data, _, meta = separate_dataset
    _arrays(data, meta)
    p = data / "val_inputs.npy"
    if problem == "missing":
        p.unlink()
    elif problem == "corrupt":
        p.write_bytes(b"PRIVATE_BROKEN_ARRAY")
    elif problem == "companion_feature":
        f = pd.read_parquet(data / "val_inputs.parquet").assign(extra=1)
        f.to_parquet(data / "val_inputs.parquet", index=False)
    elif problem == "temporal_without_time":
        meta.update(split="temporal", time_col="time")
        _write_meta(data, meta)
    else:
        arr = {"short": np.zeros((3, 2, 3), dtype="float32"), "shape": np.zeros((4, 6), dtype="float32"),
               "dtype": np.zeros((4, 2, 3), dtype="complex128"), "nan": np.full((4, 2, 3), np.nan),
               "object": np.array([object()] * 4), "zero_width": np.zeros((4, 0)), "scalar": np.array(2)}[problem]
        np.save(p, arr)
    report = validators.validate(ws)
    assert not report.ok, report.messages
    assert "PRIVATE_BROKEN_ARRAY" not in report.model_dump_json()
    if problem in {"missing", "corrupt"}:
        assert not report.checks["files_present"]


def test_array_target_copy_is_id_aligned(separate_dataset):
    ws, data, hidden, meta = separate_dataset
    _arrays(data, meta)
    for split in ("train", "val", "test"):
        p = (data if split == "train" else hidden) / f"{split}_labels.parquet"
        f = pd.read_parquet(p)
        np.save(data / f"{split}_inputs.npy", np.repeat(f["y"].to_numpy()[:, None], 3, axis=1))
        f.iloc[::-1].to_parquet(p, index=False)
    report = validators.validate(ws)
    assert report.checks["format"] and not report.checks["no_label_leak"]
    assert "ID-aligned target" in report.data_card


@pytest.mark.parametrize("problem", [None, "unknown_index", "fractional", "name", "missing_class",
                                      "duplicate_classes", "invalid_classes"])
def test_multiclass_encoding(separate_dataset, problem):
    ws, data, hidden, meta = separate_dataset
    ws = ws.model_copy(update={"spec": ws.spec.model_copy(update={"prediction_format": "class_probs",
                                                              "primary_metric": "macro_f1", "metrics": ["macro_f1"]})})
    meta.update(prediction_format="class_probs", classes=["normal", "fault"])
    for split in ("train", "val", "test"):
        p = (data if split == "train" else hidden) / f"{split}_labels.parquet"
        f = pd.read_parquet(p)
        f["y"] = [0, 1, 0, 1]
        if split == "val" and problem in {"unknown_index", "fractional", "name", "missing_class"}:
            f["y"] = {"unknown_index": [0, 2, 0, 1], "fractional": [0., .5, 0., 1.],
                      "name": ["normal", "fault", "normal", "fault"], "missing_class": [0, 0, 0, 0]}[problem]
        f.to_parquet(p, index=False)
    if problem == "duplicate_classes":
        meta["classes"] = ["normal", "normal"]
    elif problem == "invalid_classes":
        meta["classes"] = [{"secret": "PRIVATE_LABEL"}, "fault"]
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert report.ok == (problem is None), report.messages
    assert "PRIVATE_LABEL" not in report.model_dump_json()


@pytest.mark.parametrize("problem", [None, "absolute", "traversal", "non_neutral", "missing", "corrupt", "signal", "symlink"])
def test_file_images_integrity_and_pixel_target_scan(separate_dataset, tmp_path, problem):
    Image = pytest.importorskip("PIL.Image")
    ws, data, _, meta = separate_dataset
    meta["input_kind"] = dict.fromkeys(("train", "val", "test"), "files")
    for split in ("train", "val", "test"):
        f = pd.read_parquet(data / f"{split}_inputs.parquet")[["sample_id"]]
        paths = []
        for sid in f["sample_id"]:
            rel = Path("files") / split / f"{sid}.png"
            p = data / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (len(paths), ("train", "val", "test").index(split), 0)).save(p)  # distinct across splits
            paths.append(str(rel))
        f["path"] = paths
        if split == "val":
            if problem == "absolute":
                f.loc[0, "path"] = str(data / paths[0])
            elif problem == "traversal":
                f.loc[0, "path"] = "../../secret.png"
            elif problem == "non_neutral":
                f.loc[0, "path"] = "files/val/fault.png"
            elif problem == "missing":
                (data / paths[0]).unlink()
            elif problem == "corrupt":
                (data / paths[0]).write_bytes(b"PRIVATE_IMAGE_PAYLOAD")
            elif problem == "signal":
                f.loc[0, "path"] = paths[0].replace(".png", ".mat")
            elif problem == "symlink":
                external = tmp_path / "external.png"
                Image.new("RGB", (4, 4)).save(external)
                (data / paths[0]).unlink()
                (data / paths[0]).symlink_to(external)
        f.to_parquet(data / f"{split}_inputs.parquet", index=False)
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert report.ok == (problem is None), report.messages
    assert report.checks["no_label_leak"] == (problem is None)
    assert report.checks["format"] == (problem is None), report.messages
    assert "PRIVATE_IMAGE_PAYLOAD" not in report.model_dump_json()
    if problem in {"missing", "corrupt"}:
        assert not report.checks["files_present"]


@pytest.mark.parametrize("kind, fmt", [("table", "scalar"), ("array", "scalar"), ("array", "class_probs"),
                                      ("table", "binary_score"), ("array", "binary_score")])
def test_real_writer_runner_and_scorer_without_validator_stub(dataset, monkeypatch, kind, fmt):
    from ml import runner
    ws = dataset["ws"].model_copy(update={"spec": dataset["ws"].spec.model_copy(update={
        "prediction_format": fmt,
        "primary_metric": {"class_probs": "accuracy", "binary_score": "auroc", "scalar": "mae"}[fmt],
        "metrics": [{"class_probs": "accuracy", "binary_score": "auroc", "scalar": "mae"}[fmt]],
    })})
    ws = workspace.create_workspace(ws.run_id, ws.spec)
    code = '''
import harness, numpy as np, pandas as pd
parts = []
for i in range(3):
    X = np.arange(24, dtype="float32").reshape(4, 2, 3) if KIND == "array" else pd.DataFrame({"x": [2.,4.,6.,8.]})
    y = ["fault", "normal", "fault", "normal"] if FMT == "class_probs" else [1.,3.,5.,7.]
    if FMT == "binary_score":
        y = [0,1,0,1]
    parts.append((X, y))
harness.write_dataset(*parts, groups={s: [i]*4 for i,s in enumerate(("train","val","test"))},
                      meta={"split": "group", "group_col": "entity"})
'''.replace("KIND", repr(kind)).replace("FMT", repr(fmt))
    report = runner.run_prepare(ws, code)
    assert report.ok, (report.messages, report.log_tail)
    assert report.checks["prepare_runs"]
    assert (Path(ws.path) / "data_card.md").exists()
    train = '''
import harness, os, json, numpy as np, pandas as pd
from pathlib import Path
X, y = harness.load_train()
assert len(X) == len(y) == len(harness.ids("train"))
assert harness.groups("train").index.tolist() == harness.ids("train")
exp = Path(os.environ["ASSEMBLY_EXP_DIR"])
for split, Xs in (("val", harness.load_val_inputs()), ("test", harness.load_test_inputs())):
    assert len(Xs) == len(harness.ids(split))
    out = {"sample_id": harness.ids(split)}
    if FMT == "class_probs":
        assert harness.classes() == ["fault", "normal"]
        out.update(p_0=[1.,0.,1.,0.], p_1=[0.,1.,0.,1.])
    elif FMT == "binary_score":
        out["score"] = [0.1,0.9,0.1,0.9]
    else:
        out["pred"] = [1.,3.,5.,7.]
    pd.DataFrame(out).iloc[::-1].to_parquet(exp / f"{split}_predictions.parquet", index=False)
(exp / "timings.json").write_text(json.dumps({"p50_latency_ms": 0.1, "p95_latency_ms": 0.2}))
(exp / "predict_meta.json").write_text(json.dumps({"threshold": 0.5}))
'''.replace("FMT", repr(fmt))
    result = runner.run_experiment(ws, train, {"description": "real validator integration"})
    assert result.status == "keep", result.log_tail
    metric = {"class_probs": "accuracy", "binary_score": "auroc", "scalar": "mae"}[fmt]
    assert result.metrics[metric] == (0.0 if fmt == "scalar" else 1.0)
    assert workspace.best_result(ws).exp_id == result.exp_id
    final = runner.final_test(ws)
    assert final.test_metrics[metric] == result.metrics[metric]


def test_real_runner_export_package_with_real_validator(dataset):
    from ml import package, runner

    pytest.importorskip("sklearn")
    ws = workspace.create_workspace(dataset["ws"].run_id, dataset["ws"].spec)
    prepare = '''
import os,json,pandas as pd
from pathlib import Path
parts = ("train","val","test")
data, hidden = Path("data"), Path(os.environ["ASSEMBLY_LABELS_DIR"])
for split in parts:
    ids = [f"{split}-{i:06d}" for i in range(4)]
    pd.DataFrame({"sample_id": ids, "x": [1.,2.,3.,4.]}).to_parquet(data / f"{split}_inputs.parquet", index=False)
    pd.DataFrame({"sample_id": ids, "y": [3.,5.,7.,9.]}).to_parquet(
        (data if split == "train" else hidden) / f"{split}_labels.parquet", index=False)
(data / "meta.json").write_text(json.dumps({"prediction_format": "scalar", "split": "group", "group_col": "entity",
    "input_kind": dict.fromkeys(parts,"table"), "sizes": dict.fromkeys(parts,4),
    "groups": {s:[i]*4 for i,s in enumerate(parts)}}))
'''
    report = runner.run_prepare(ws, prepare)
    assert report.ok, (report.messages, report.log_tail)
    train = '''
import os,json,pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from ml.models.export import save_model
exp = Path(os.environ["ASSEMBLY_EXP_DIR"])
X = pd.read_parquet("data/train_inputs.parquet").set_index("sample_id")
y = pd.read_parquet("data/train_labels.parquet").set_index("sample_id")["y"]
model = LinearRegression().fit(X,y)
save_model(model,str(exp / "model"),example_input=X.iloc[:4])
for split in ("val","test"):
    Xs = pd.read_parquet(f"data/{split}_inputs.parquet").set_index("sample_id")
    pd.DataFrame({"sample_id": Xs.index, "pred": model.predict(Xs)}).to_parquet(exp / f"{split}_predictions.parquet",index=False)
(exp / "timings.json").write_text(json.dumps({"p50_latency_ms":0.1,"p95_latency_ms":0.2}))
'''
    result = runner.run_experiment(ws, train, {})
    assert result.status == "keep", result.log_tail
    runner.final_test(ws)
    predictor = package.load_predictor(package.package(ws))
    assert predictor.predict({"x": 2.})["value"] == pytest.approx(5.)
    assert predictor.predict({"replay": 0})["value"] == pytest.approx(3.)


def test_array_companion_reordering_is_rejected(separate_dataset):
    ws, data, _, meta = separate_dataset
    _arrays(data, meta)
    p = data / "val_inputs.parquet"
    pd.read_parquet(p).iloc[::-1].to_parquet(p, index=False)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["format"]
    assert "ordinal IDs" in report.data_card


def test_compatible_real_array_dtypes_and_one_dimensional_inputs(separate_dataset):
    ws, data, _, meta = separate_dataset
    _arrays(data, meta)
    for split, dtype in zip(("train", "val", "test"), ("int32", "float32", "float64"), strict=True):
        np.save(data / f"{split}_inputs.npy", np.array([2, 4, 6, 8], dtype=dtype))
    report = validators.validate(ws)
    assert report.ok, report.messages


def test_new_hidden_targets_and_errors_are_not_disclosed(separate_dataset):
    ws, _data, hidden, _ = separate_dataset
    p = hidden / "val_labels.parquet"
    f = pd.read_parquet(p).assign(y="PRIVATE_INVALID_TARGET")
    f.to_parquet(p, index=False)
    report = validators.validate(ws)
    assert not report.ok and "PRIVATE_INVALID_TARGET" not in report.model_dump_json()


def test_event_time_is_forbidden_as_an_input(separate_dataset):
    ws, data, _, _ = separate_dataset
    for split in ("train", "val", "test"):
        p = data / f"{split}_inputs.parquet"
        pd.read_parquet(p).assign(event_time=100).to_parquet(p, index=False)
    assert not validators.validate(ws).checks["no_label_leak"]


@pytest.mark.parametrize("problem", ["duplicate_inputs", "duplicate_labels", "multilevel"])
def test_malformed_new_columns_do_not_crash(separate_dataset, monkeypatch, problem):
    ws, _, _, _ = separate_dataset
    real_read = pd.read_parquet

    def read(path, **kwargs):
        f = real_read(path, **kwargs)
        if path.name == ("train_labels.parquet" if problem == "duplicate_labels" else "train_inputs.parquet"):
            if problem == "multilevel":
                f.columns = pd.MultiIndex.from_tuples([(c, "value") for c in f.columns])
            else:
                f = pd.concat([f, f[["sample_id"]]], axis=1)
        return f

    monkeypatch.setattr(pd, "read_parquet", read)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["format"]


def test_new_metadata_diagnostics_do_not_dump_arbitrary_values(separate_dataset):
    ws, data, _, meta = separate_dataset
    meta["input_kind"]["val"] = ["PRIVATE_ENCODING_VALUE"]
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok and "PRIVATE_ENCODING_VALUE" not in report.model_dump_json()


@pytest.mark.parametrize("copy_target", [False, True])
def test_explicit_group_input_remains_a_feature(separate_dataset, copy_target):
    ws, data, hidden, meta = separate_dataset
    for split in ("train", "val", "test"):
        p = data / f"{split}_inputs.parquet"
        pd.read_parquet(p).assign(group=meta["groups"][split]).to_parquet(p, index=False)
        if copy_target:
            p = (data if split == "train" else hidden) / f"{split}_labels.parquet"
            pd.read_parquet(p).assign(y=meta["groups"][split]).to_parquet(p, index=False)
    report = validators.validate(ws)
    assert report.checks["format"], report.messages
    assert report.ok == (not copy_target)
    assert report.checks["no_label_leak"] == (not copy_target)
    assert "`group`:" in report.data_card


def test_invalid_metadata_groups_cannot_be_replaced_by_input_groups(separate_dataset):
    ws, data, _, meta = separate_dataset
    for split in ("train", "val", "test"):
        p = data / f"{split}_inputs.parquet"
        pd.read_parquet(p).assign(group=meta["groups"][split]).to_parquet(p, index=False)
    meta["groups"]["val"] = [[2]] * 4
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["group_disjoint"]


def test_new_groups_do_not_require_an_optional_source_column_name(separate_dataset):
    ws, data, _, meta = separate_dataset
    meta.pop("group_col")
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert report.ok, report.messages


def test_empty_array_split_fails_without_crashing_the_leak_scan(separate_dataset):
    ws, data, hidden, meta = separate_dataset
    _arrays(data, meta)
    for p in (data / "val_inputs.parquet", hidden / "val_labels.parquet"):
        pd.read_parquet(p).iloc[:0].to_parquet(p, index=False)
    np.save(data / "val_inputs.npy", np.zeros((0, 2, 3), dtype="float32"))
    meta["groups"]["val"] = []
    meta["sizes"]["val"] = 0
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["min_sizes"] and not report.checks["no_label_leak"]


@pytest.mark.parametrize("artifact", ["train_inputs.parquet", "train_labels.parquet", "val_labels.parquet",
                                      "meta.json", "val_inputs.npy"])
@pytest.mark.parametrize("redirect", ["outside", "inside", "loop"])
def test_dataset_artifact_symlinks_are_rejected(separate_dataset, tmp_path, artifact, redirect):
    ws, data, hidden, meta = separate_dataset
    if artifact.endswith(".npy"):
        _arrays(data, meta)
    path = (hidden if artifact == "val_labels.parquet" else data) / artifact
    payload = path.read_bytes()
    path.unlink()
    target = path if redirect == "loop" else (data if redirect == "inside" else tmp_path) / "redirect"
    if redirect != "loop":
        target.write_bytes(payload)
    path.symlink_to(target)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["no_label_leak"], report.messages
    assert "PRIVATE" not in report.model_dump_json()


@pytest.mark.parametrize("fmt", ["boxes", "masks"])
def test_real_structured_labels_are_explicitly_unverified(separate_dataset, fmt):
    ws, data, hidden, meta = separate_dataset
    ws = ws.model_copy(update={"spec": ws.spec.model_copy(update={"prediction_format": fmt})})
    meta["prediction_format"] = fmt
    for split in ("train", "val", "test"):
        path = (data if split == "train" else hidden) / f"{split}_labels.parquet"
        labels = pd.read_parquet(path)
        path.unlink()
        path.with_suffix(".json").write_text(json.dumps({sid: [] for sid in labels["sample_id"]}))
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok
    assert not report.checks["label_sanity"] and not report.checks["no_label_leak"]
    assert "structured labels are not validated" in report.data_card


def test_train_label_duplicates_are_not_hidden_by_a_merge(separate_dataset):
    ws, data, _, _ = separate_dataset
    path = data / "train_labels.parquet"
    labels = pd.read_parquet(path)
    pd.concat([labels, labels.iloc[:1]]).to_parquet(path, index=False)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["format"] and not report.checks["no_label_leak"]


@pytest.mark.parametrize("normal_class", [None, "normal", "fault", "PRIVATE_UNKNOWN_CLASS", 0, []])
def test_normal_class_must_match_the_packaged_alarm_vocabulary(separate_dataset, normal_class):
    ws, data, hidden, meta = separate_dataset
    ws = ws.model_copy(update={"spec": ws.spec.model_copy(update={"prediction_format": "class_probs",
                                                              "primary_metric": "accuracy", "metrics": ["accuracy"]})})
    meta.update(prediction_format="class_probs", classes=["normal", "fault"], normal_class=normal_class)
    for split in ("train", "val", "test"):
        path = (data if split == "train" else hidden) / f"{split}_labels.parquet"
        pd.read_parquet(path).assign(y=[0, 1, 0, 1]).to_parquet(path, index=False)
    _write_meta(data, meta)
    report = validators.validate(ws)
    valid = normal_class is None or normal_class in ["normal", "fault"]
    assert report.ok == valid, report.messages
    assert report.checks["label_sanity"] == valid
    assert "PRIVATE_UNKNOWN_CLASS" not in report.model_dump_json()


@pytest.mark.parametrize("problem", ["copied_pixels", "shape", "nonfinite", "numeric_metadata", "group_overlap",
                                      "embedded_target"])
def test_image_checks_reject_real_leakage_and_unverifiable_content(separate_dataset, problem):
    from PIL import Image

    ws, data, hidden, meta = separate_dataset
    meta["input_kind"] = dict.fromkeys(("train", "val", "test"), "files")
    for phase, split in enumerate(("train", "val", "test")):
        path = data / f"{split}_inputs.parquet"
        frame = pd.read_parquet(path)[["sample_id"]]
        label_path = (data if split == "train" else hidden) / f"{split}_labels.parquet"
        labels = pd.read_parquet(label_path).set_index("sample_id")["y"]
        paths = []
        for i, sid in enumerate(frame["sample_id"]):
            pixels = np.full((4, 4), 20 + 10 * phase + i, dtype="float32")
            if problem == "copied_pixels":
                pixels[0, 0] = labels.loc[sid]
            elif split == "val" and i == 0:
                if problem == "shape":
                    pixels = np.zeros((5, 4), dtype="float32")
                elif problem == "nonfinite":
                    pixels[0, 0] = np.nan
            suffix = ".png" if problem == "embedded_target" else ".tif"
            rel = Path("files") / split / f"{sid}{suffix}"
            (data / rel).parent.mkdir(parents=True, exist_ok=True)
            if problem == "embedded_target":
                from PIL.PngImagePlugin import PngInfo

                info = PngInfo()
                info.add_text("label", str(labels.loc[sid]))
                Image.fromarray(pixels.astype("uint8")).save(data / rel, pnginfo=info)
            else:
                Image.fromarray(pixels).save(data / rel)
            paths.append(str(rel))
        frame["path"] = paths
        if problem == "numeric_metadata":
            frame["temperature"] = labels.loc[frame["sample_id"]].to_numpy()
        frame.to_parquet(path, index=False)
        labels.iloc[::-1].reset_index().to_parquet(label_path, index=False)
    if problem == "group_overlap":
        meta["groups"]["val"] = meta["groups"]["train"]
    _write_meta(data, meta)
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    report = validators.validate(ws)
    assert not report.ok, report.messages
    if problem == "group_overlap":
        assert not report.checks["group_disjoint"]
    else:
        assert not report.checks["no_label_leak"]
    if problem in {"copied_pixels", "numeric_metadata"}:
        assert "ID-aligned target" in report.data_card
    elif problem == "shape":
        assert "NOT VERIFIED" in report.data_card
    assert before == {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}


@pytest.mark.parametrize("split", ["val", "test"])
def test_public_metadata_cannot_embed_hidden_label_tables(separate_dataset, split):
    ws, data, hidden, meta = separate_dataset
    labels = pd.read_parquet(hidden / f"{split}_labels.parquet")
    labels["y"] = [91427.123456, 82538.234567, 73649.345678, 64750.456789]
    labels.to_parquet(hidden / f"{split}_labels.parquet", index=False)
    meta[f"{split}_labels"] = labels.set_index("sample_id")["y"].to_dict()
    _write_meta(data, meta)
    report = validators.validate(ws)
    assert not report.ok and not report.checks["no_label_leak"], report.messages
    assert all(str(v) not in report.model_dump_json() for v in labels["y"])
