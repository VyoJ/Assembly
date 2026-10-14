"""Real validator tests against synthetic persisted Parquet tables."""

import builtins
import hashlib
import json

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
