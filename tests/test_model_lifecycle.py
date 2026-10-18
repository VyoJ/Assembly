"""Real writer, lifecycle, restoration and CPU runner integration; no lifecycle/validator stubs."""

import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import package, runner, scoring, validators, workspace
from ml.models import export


def _spec(fmt="scalar"):
    metric = {"scalar": "mae", "binary_score": "auroc", "class_probs": "accuracy",
              "boxes": "map50", "masks": "iou"}[fmt]
    return TaskSpec(
        id="lifecycle", name="Industrial CPU lifecycle", rationale="Check integration", framing="Synthetic sensors",
        input_streams=[], label_sources=[], target_definition="Synthetic target", prediction_format=fmt,
        primary_metric=metric, metrics=[metric, "p95_latency_ms"],
        kpis=[KPI(metric=metric, op="<=" if fmt == "scalar" else ">=", threshold=1e-5 if fmt == "scalar" else .9)],
        split_strategy="by machine", time_budget_s=30,
    )


def _snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")

    def make(kind="table", fmt="scalar"):
        ws = workspace.create_workspace("lifecycle-unit", _spec(fmt))
        hidden = workspace.labels_dir(ws.run_id, ws.task_id)
        exp = Path(ws.path) / "exp_local"
        monkeypatch.setenv("ASSEMBLY_WORKSPACE", ws.path)
        monkeypatch.setenv("ASSEMBLY_EXP_DIR", str(exp))
        monkeypatch.setenv("ASSEMBLY_LABELS_DIR", str(hidden))
        module_spec = importlib.util.spec_from_file_location("lifecycle_harness", Path(ws.path) / "harness.py")
        h = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(h)
        rng = np.random.default_rng(14)
        parts = []
        for split, n in zip(("train", "val", "test"), (36, 9, 9), strict=True):
            raw = rng.normal(size=(n, 6))
            if fmt == "scalar":
                y = 5 + raw[:, 0] * 3 - raw[:, 1]
            elif fmt == "binary_score":
                y = np.arange(n) % 2
                raw[:, 0] = 4 * y - 2 + rng.normal(scale=.1, size=n)
            elif fmt == "class_probs":
                indices = np.arange(n) % 3
                raw[np.arange(n), indices] += 5
                y = ["normal", "fault_a", "fault_b"]
                y = [y[i] for i in indices]
            else:
                y = [[[0, 0, 2, 2, 0]]] * n if fmt == "boxes" else [None] * n
            if kind == "table":
                X = pd.DataFrame(raw, columns=[f"s{i}" for i in range(6)])
            elif kind == "files":
                from PIL import Image

                X = []
                for i in range(n):
                    path = tmp_path / f"{split}_{i}.png"
                    img = Image.new("L", (4, 4), color=(i % 2) * 255)
                    img.putpixel((3, 2), i + 1)  # distinct files: identical content across splits fails validation
                    img.putpixel((3, 3), 100 + ("train", "val", "test").index(split))
                    img.save(path)
                    X.append(str(path))
            else:
                X = raw.astype("float32")
                if kind == "tensor":
                    X = X.reshape(n, 2, 3)
            parts.append((X, y))
        h.write_dataset(*parts, groups={s: [s] * len(X) for s, (X, _) in zip(
            ("train", "val", "test"), parts, strict=True)}, meta={"split": "group"})
        monkeypatch.delenv("ASSEMBLY_LABELS_DIR")
        return h, ws, hidden

    return make


def _fit(h, kind, fmt):
    X, y = h.load_train()
    steps = [StandardScaler(), LinearRegression() if fmt == "scalar" else LogisticRegression(max_iter=200)]
    if kind == "tensor":
        steps.insert(0, FunctionTransformer(lambda W: W.reshape(len(W), -1)))
    model = make_pipeline(*steps).fit(X, y)
    return model


@pytest.mark.parametrize("kind", ["table", "array", "tensor"])
@pytest.mark.parametrize("fmt", ["scalar", "binary_score", "class_probs"])
def test_save_restore_in_a_fresh_process_with_preprocessing(lifecycle, tmp_path, kind, fmt):
    h, ws, hidden = lifecycle(kind, fmt)
    model = _fit(h, kind, fmt)
    before = _snapshot(h.DATA_DIR), _snapshot(hidden)
    X = h.load_val_inputs()
    original = X.copy()
    h.save_model(model, threshold=.4, purpose="regression check")
    path = h.EXP_DIR / "model"
    info = json.loads((path / "model.json").read_text())
    assert info["prediction_format"] == fmt and info["threshold"] == .4 and info["purpose"] == "regression check"
    assert info["input_kind"] == ("table" if kind == "table" else "array")
    if kind == "table":
        assert info["feature_columns"] == list(X.columns)
        assert info["feature_dtypes"] == {c: str(t) for c, t in X.dtypes.items()}
        assert len(h._example_input(X)) == 8
    else:
        assert info["input_shape"] == list(X.shape[1:]) and info["input_dtype"] == "float32"
        assert h._example_input(X).shape == (8, *X.shape[1:])
    if fmt == "class_probs":
        assert info["classes"] == h.classes()
    restored = export.load_saved(str(path))
    expected = model.predict(X) if fmt == "scalar" else model.predict_proba(X)
    actual = restored.predict(X) if fmt == "scalar" else restored.predict_proba(X)
    np.testing.assert_allclose(actual, expected)
    script = '''
import sys, numpy as np, pandas as pd
from ml.models.export import load_saved
model, data, kind, fmt, out = sys.argv[1:]
X = pd.read_parquet(data + '/val_inputs.parquet').set_index('sample_id') if kind == 'table' else np.load(data + '/val_inputs.npy')
saved = load_saved(model)
np.save(out, saved.predict(X) if fmt == 'scalar' else saved.predict_proba(X))
'''
    out = tmp_path / "restored.npy"
    proc = subprocess.run([sys.executable, "-c", script, str(path), str(h.DATA_DIR), kind, fmt, str(out)],
                          check=False, text=True, capture_output=True, cwd=ws.path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    np.testing.assert_allclose(np.load(out), expected)
    if kind == "table":
        pd.testing.assert_frame_equal(X, original)
    else:
        np.testing.assert_array_equal(X, original)
    assert before == (_snapshot(h.DATA_DIR), _snapshot(hidden))


def test_nonnumeric_dataframe_pipeline_preserves_schema(lifecycle):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    h, _, _ = lifecycle()
    for split in ("train", "val", "test"):
        path = h.DATA_DIR / f"{split}_inputs.parquet"
        pd.read_parquet(path).assign(mode="running").to_parquet(path, index=False)
    X, y = h.load_train()
    transform = ColumnTransformer([("mode", OneHotEncoder(handle_unknown="ignore"), ["mode"]),
                                   ("numeric", StandardScaler(), [f"s{i}" for i in range(6)])])
    model = make_pipeline(transform, LinearRegression()).fit(X, y)
    assert h._example_input(X) is None
    h.save_model(model)
    saved = export.load_saved(str(h.EXP_DIR / "model"))
    assert saved.info["feature_columns"] == list(X.columns)
    np.testing.assert_allclose(saved.predict(X), model.predict(X))


@pytest.mark.parametrize("invalid", [None, 1, "model", [], object()])
def test_invalid_models_leave_a_previous_artifact_intact(lifecycle, invalid):
    h, _, _ = lifecycle()
    h.save_model(_fit(h, "table", "scalar"))
    before = _snapshot(h.EXP_DIR)
    with pytest.raises(TypeError, match="requires a model"):
        h.save_model(invalid)
    assert _snapshot(h.EXP_DIR) == before


def test_failed_real_serialization_does_not_corrupt_the_saved_model(lifecycle):
    class BrokenModel:
        def predict(self, X):
            return np.zeros(len(X))

        def __reduce__(self):
            raise RuntimeError("serializer failure")

    h, _, _ = lifecycle()
    h.save_model(_fit(h, "table", "scalar"))
    before = _snapshot(h.EXP_DIR)
    with pytest.raises(RuntimeError, match="serializer failure"):
        h.save_model(BrokenModel())
    assert _snapshot(h.EXP_DIR) == before


@pytest.mark.parametrize("failure", [None, "publication", "rollback"])
def test_model_publication_and_rollback_preserve_artifact_integrity(lifecycle, monkeypatch, failure):
    h, _, _ = lifecycle()
    model = _fit(h, "table", "scalar")
    h.save_model(model)
    previous = _snapshot(h.EXP_DIR / "model")
    rename = Path.rename

    def denied(source, target):
        if (Path(target) == h.EXP_DIR / "model" and failure is not None
                and (failure == "rollback" or source.name == "model")):
            raise PermissionError("publication and rollback denied")
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", denied)
    if failure is None:
        h.save_model(model, marker="updated")
        assert json.loads((h.EXP_DIR / "model" / "model.json").read_text())["marker"] == "updated"
        assert {p.name for p in h.EXP_DIR.iterdir()} == {"model"}
        return
    with pytest.raises(OSError) as error:
        h.save_model(model)
    recoveries = [p.parent for p in h.EXP_DIR.rglob("model.json")]
    assert len(recoveries) == 1
    assert _snapshot(recoveries[0]) == previous
    if failure == "rollback":
        assert str(recoveries[0]) in str(error.value)
    else:
        assert recoveries[0] == h.EXP_DIR / "model"


@pytest.mark.parametrize("file, corrupt", [("model.json", False), ("model.json", True),
                                           ("model.joblib", False), ("model.joblib", True)])
def test_missing_and_corrupted_saved_artifacts_fail(lifecycle, file, corrupt):
    h, _, _ = lifecycle()
    h.save_model(_fit(h, "table", "scalar"))
    path = h.EXP_DIR / "model" / file
    path.write_bytes(b"invalid artifact") if corrupt else path.unlink()
    with pytest.raises((FileNotFoundError, ValueError, KeyError, pickle.UnpicklingError, IndexError, ImportError)):
        export.load_saved(str(path.parent))


def test_torch_cpu_export_restore(lifecycle, tmp_path):
    import torch

    h, _, _ = lifecycle("array")
    model = torch.nn.Linear(6, 1).eval()
    X = h.load_val_inputs()
    with torch.no_grad():
        expected = model(torch.from_numpy(X)).numpy()
    h.save_model(model)
    saved = export.load_saved(str(h.EXP_DIR / "model"))
    assert saved.kind in {"onnx", "torchscript"}
    np.testing.assert_allclose(saved.predict(X), expected, rtol=1e-5, atol=1e-6)
    script = '''
import sys, numpy as np
from ml.models.export import load_saved
np.save(sys.argv[3], load_saved(sys.argv[1]).predict(np.load(sys.argv[2])))
'''
    out = tmp_path / "torch-restored.npy"
    proc = subprocess.run([sys.executable, "-c", script, str(h.EXP_DIR / "model"),
                           str(h.DATA_DIR / "val_inputs.npy"), str(out)],
                          check=False, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    np.testing.assert_allclose(np.load(out), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["table", "array", "tensor"])
@pytest.mark.parametrize("fmt", ["scalar", "binary_score", "class_probs"])
def test_finish_real_predictions_ids_schemas_scoring_and_label_privacy(lifecycle, monkeypatch, kind, fmt):
    h, ws, hidden = lifecycle(kind, fmt)
    model = _fit(h, kind, fmt)
    predict = model.predict if fmt == "scalar" else model.predict_proba
    before = _snapshot(h.DATA_DIR), _snapshot(hidden)
    read = pd.read_parquet

    def guarded(path, *args, **kwargs):
        assert not Path(path).resolve().is_relative_to(hidden.resolve()), "finish accessed hidden labels"
        return read(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("finish must use input-only loaders")

    with monkeypatch.context() as guard:
        guard.setattr(pd, "read_parquet", guarded)
        guard.setattr(h, "load_train", forbidden)
        guard.setattr(h, "read_labels", forbidden)
        h.finish(predict, threshold=.4 if fmt == "binary_score" else None)
    assert {p.name for p in h.EXP_DIR.iterdir()} == {
        "val_predictions.parquet", "test_predictions.parquet", "predict_meta.json", "timings.json"}
    for split in ("val", "test"):
        df = pd.read_parquet(h.EXP_DIR / f"{split}_predictions.parquet")
        columns = {"scalar": ["pred"], "binary_score": ["score"], "class_probs": ["p_0", "p_1", "p_2"]}[fmt]
        assert list(df.columns) == ["sample_id", *columns] and df["sample_id"].tolist() == h.ids(split)
        X = h.load_val_inputs() if split == "val" else h.load_test_inputs()
        expected = predict(X)
        if fmt == "binary_score":
            expected = expected[:, 1]
        np.testing.assert_allclose(df[columns].to_numpy(), np.asarray(expected).reshape(len(X), -1))
        metric = ws.spec.primary_metric
        scores = scoring.compute_metrics(fmt, str(h.EXP_DIR / f"{split}_predictions"),
                                         str(hidden / f"{split}_labels"), [metric])
        assert scores[metric] < 1e-5 if fmt == "scalar" else scores[metric] >= .9
    timings = json.loads((h.EXP_DIR / "timings.json").read_text())
    assert 0 < timings["p50_latency_ms"] <= timings["p95_latency_ms"]
    assert timings["latency_samples"] == 9 and all(t > 0 for t in timings["batch_inference_ms"].values())
    assert "input slicing excluded" in timings["measurement"]
    assert before == (_snapshot(h.DATA_DIR), _snapshot(hidden))


def test_id_labeled_predictions_are_aligned_and_probability_columns_ordered(lifecycle):
    h, _, _ = lifecycle()
    h.finish(lambda X: pd.Series(X["s0"].to_numpy(), index=X.index).iloc[::-1])
    result = pd.read_parquet(h.EXP_DIR / "val_predictions.parquet")
    np.testing.assert_allclose(result["pred"], h.load_val_inputs()["s0"])


def test_class_dataframe_ids_and_columns_are_aligned(lifecycle):
    h, _, _ = lifecycle(fmt="class_probs")

    def predict(X):
        out = pd.DataFrame({"p_2": .2, "p_1": .3, "p_0": .5}, index=X.index)
        out.index.name = "sample_id"
        return out.iloc[::-1].reset_index()

    h.finish(predict)
    result = pd.read_parquet(h.EXP_DIR / "val_predictions.parquet")
    assert list(result.columns) == ["sample_id", "p_0", "p_1", "p_2"]
    np.testing.assert_allclose(result.iloc[:, 1:], np.tile([.5, .3, .2], (9, 1)))


@pytest.mark.parametrize("problem", ["short", "matrix", "row_vector", "rank3", "nan", "inf", "strings",
                                      "complex", "bool", "none", "error", "duplicate_ids", "wrong_ids"])
def test_invalid_scalar_outputs_publish_nothing(lifecycle, problem):
    h, _, _ = lifecycle()

    def predict(X):
        n = len(X)
        values = {"short": np.zeros(n-1), "matrix": np.zeros((n, 2)), "row_vector": np.zeros((1, n)),
                  "rank3": np.zeros((n, 1, 1)), "nan": np.full(n, np.nan), "inf": np.full(n, np.inf),
                  "strings": ["1.5"] * n, "complex": np.ones(n, dtype=complex), "bool": np.ones(n, dtype=bool),
                  "none": None, "error": {"error": "predictor failure"}}
        if problem in {"duplicate_ids", "wrong_ids"}:
            index = [X.index[0]] * n if problem == "duplicate_ids" else [f"wrong-{i}" for i in range(n)]
            return pd.Series(np.zeros(n), index=index)
        return values[problem]

    with pytest.raises(ValueError):
        h.finish(predict)
    assert not h.EXP_DIR.exists()


@pytest.mark.parametrize("fmt, output", [
    ("binary_score", [-.01]), ("binary_score", [1.01]), ("binary_score", [[.8, .8]]),
    ("binary_score", [[-.1, .9]]), ("class_probs", [[.2, .3, .1]]),
    ("class_probs", [[.2, -.1, .9]]), ("class_probs", [[.5, .5]]),
    ("class_probs", [[.5, .5, np.nan]]),
])
def test_invalid_probabilities_publish_nothing(lifecycle, fmt, output):
    h, _, _ = lifecycle(fmt=fmt)
    with pytest.raises(ValueError):
        h.finish(lambda X: np.tile(output, (len(X), 1)))
    assert not h.EXP_DIR.exists()


@pytest.mark.parametrize("classes", [None, 4, True, {}, ["normal"], ["same", "same", "other"], ["a", "b", np.nan]])
def test_invalid_class_vocabularies_are_rejected(lifecycle, classes):
    h, _, _ = lifecycle(fmt="class_probs")
    meta = h._meta()
    meta["classes"] = classes
    (h.DATA_DIR / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="vocabulary"):
        h.finish(lambda X: pd.DataFrame(np.tile([.2, .3, .5], (len(X), 1)), index=X.index))


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -.1, 1.1, True, "0.5"])
def test_invalid_thresholds_are_rejected(lifecycle, threshold):
    h, _, _ = lifecycle(fmt="binary_score")
    with pytest.raises(ValueError, match="threshold"):
        h.finish(lambda X: np.zeros(len(X)), threshold=threshold)
    assert not h.EXP_DIR.exists()


@pytest.mark.parametrize("problem", ["duplicates", "missing", "blank", "null", "overlap", "input_order"])
@pytest.mark.parametrize("kind", ["table", "array"])
def test_authoritative_input_id_errors_are_rejected(lifecycle, monkeypatch, problem, kind):
    h, ws, _ = lifecycle(kind)
    if problem == "input_order":
        if kind == "array":
            # Array content identity is unknowable; the validator certifies companion ordinal order.
            p = h.DATA_DIR / "val_inputs.parquet"
            pd.read_parquet(p).iloc[::-1].to_parquet(p, index=False)
            report = validators.validate(ws)
            assert not report.ok and not report.checks["format"]
            return
        load = h.load_val_inputs
        monkeypatch.setattr(h, "load_val_inputs", lambda: load().iloc[::-1])
    else:
        read_ids = h.ids

        def ids(split):
            sids = read_ids(split)
            if problem == "overlap" and split == "test":
                return read_ids("val")
            if split == "val":
                if problem == "duplicates":
                    sids[1] = sids[0]
                elif problem == "missing":
                    sids.pop()
                elif problem == "blank":
                    sids[0] = ""
                elif problem == "null":
                    sids[0] = None
            return sids

        monkeypatch.setattr(h, "ids", ids)
    with pytest.raises(ValueError, match="sample_ids|index order"):
        h.finish(lambda X: np.zeros(len(X)))
    assert not h.EXP_DIR.exists()


@pytest.mark.parametrize("fail_at", ["val", "test", "single"])
def test_predictor_exceptions_preserve_prior_artifacts(lifecycle, fail_at):
    h, _, _ = lifecycle()
    h.finish(lambda X: np.zeros(len(X)))
    before = _snapshot(h.EXP_DIR)

    def predict(X):
        if (fail_at == "single" and len(X) == 1 or fail_at != "single" and X.index[0].startswith(fail_at)):
            raise RuntimeError("predictor failed")
        return np.zeros(len(X))

    with pytest.raises(RuntimeError, match="predictor failed"):
        h.finish(predict)
    assert _snapshot(h.EXP_DIR) == before


def test_single_row_invalid_outputs_are_rejected(lifecycle):
    h, _, _ = lifecycle()
    with pytest.raises(ValueError, match="NaN"):
        h.finish(lambda X: np.full(len(X), np.nan if len(X) == 1 else 0.))
    assert not h.EXP_DIR.exists()


def test_actual_latency_includes_predictor_work(lifecycle):
    h, _, _ = lifecycle()

    def predict(X):
        time.sleep(.002)
        return np.zeros(len(X))

    h.finish(predict)
    timings = json.loads((h.EXP_DIR / "timings.json").read_text())
    assert timings["p50_latency_ms"] >= 1.5
    assert min(timings["batch_inference_ms"].values()) >= 1.5


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_synchronization_hook(lifecycle, monkeypatch, device):
    h, _, _ = lifecycle()
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: device == "cuda",
                                                synchronize=lambda: calls.append("cuda")),
                            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: device == "mps")),
                            mps=SimpleNamespace(synchronize=lambda: calls.append("mps")))
    monkeypatch.setitem(sys.modules, "torch", torch)
    h._sync_inference()
    assert calls == [device]


def test_file_predictor_contract(lifecycle):
    h, _, _ = lifecycle("files", "binary_score")
    h.finish(lambda X: np.array([h.load_image(p)[0, 0] / 255 for p in X["path"]]))  # an unstamped pixel
    df = pd.read_parquet(h.EXP_DIR / "val_predictions.parquet")
    assert df["score"].tolist() == [0., 1., 0., 1., 0., 1., 0., 1., 0.]


def test_equivalent_image_pixels_validate_as_files_and_arrays(lifecycle, monkeypatch):
    h, ws, _ = lifecycle("files", "binary_score")
    before = _snapshot(h.DATA_DIR)
    report = validators.validate(ws)
    assert report.ok and report.checks["no_label_leak"], report.messages
    assert _snapshot(h.DATA_DIR) == before
    parts = []
    for split in ("train", "val", "test"):
        X = h._load_inputs(split)
        parts.append((np.stack([h.load_image(p) for p in X["path"]]), np.arange(len(X)) % 2))
    monkeypatch.setenv("ASSEMBLY_LABELS_DIR", str(workspace.labels_dir(ws.run_id, ws.task_id)))
    h.write_dataset(*parts, groups={s: [s] * len(X) for s, (X, _) in zip(
        ("train", "val", "test"), parts, strict=True)}, meta={"split": "group"})
    array_report = validators.validate(ws)
    assert array_report.ok and array_report.checks == report.checks, array_report.messages


def test_group_image_dataset_lifecycle_preserves_model_preprocessing(lifecycle):
    h, ws, _ = lifecycle("files", "binary_score")
    before = _snapshot(h.DATA_DIR)
    report = validators.validate(ws)
    assert report.ok, report.messages
    train = '''
import harness, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
def images(X):
    return np.stack([harness.load_image(p) for p in X['path']])
transform = FunctionTransformer(lambda batch: np.stack([
    batch.reshape(len(batch), -1).max(axis=1), batch.reshape(len(batch), -1).mean(axis=1)], axis=1))
X,y = harness.load_train()
batch = images(X)
model = make_pipeline(transform, LogisticRegression()).fit(batch, y)
harness.save_model(model, example_input=batch[:8])
harness.finish(lambda X: model.predict_proba(images(X)), threshold=.5)
'''
    result = runner.run_experiment(ws, train, {"description": "group images with saved preprocessing"})
    assert result.status == "keep" and result.all_kpis_passed, result.log_tail
    assert runner.final_test(ws).passed
    predictor = package.load_predictor(package.package(ws))
    predictions = pd.read_parquet(Path(ws.path) / result.exp_id / "test_predictions.parquet")
    X = h.load_test_inputs()
    for i in (0, 1):
        live = predictor.predict({"image_path": X["path"].iloc[i]})
        assert live["score"] == pytest.approx(predictions["score"].iloc[i], abs=1e-4)
    assert _snapshot(h.DATA_DIR) == before


def test_box_prediction_schema_and_actual_scorer(lifecycle):
    h, _, hidden = lifecycle("array", "boxes")
    h.finish(lambda X: [[[0, 0, 2, 2, .9, 0]]] * len(X), threshold=.5)
    path = h.EXP_DIR / "val_predictions.json"
    assert list(json.loads(path.read_text())) == h.ids("val")
    metrics = scoring.compute_metrics("boxes", str(path), str(hidden / "val_labels.json"), ["map50"])
    assert metrics["map50"] == 1.


@pytest.mark.parametrize("box", [[0, 0, 2, 2, .9], [2, 2, 0, 0, .9, 0], [0, 0, 2, 2, 1.1, 0],
                                [0, 0, 2, 2, .9, .5], [0, 0, np.nan, 2, .9, 0]])
def test_invalid_boxes_are_rejected(lifecycle, box):
    h, _, _ = lifecycle("array", "boxes")
    with pytest.raises(ValueError, match="boxes"):
        h.finish(lambda X: [[box]] * len(X))
    assert not h.EXP_DIR.exists()


def test_masks_fail_explicitly_until_the_contract_is_validated(lifecycle):
    h, _, _ = lifecycle("array", "masks")
    with pytest.raises(ValueError, match="masks"):
        h.finish(lambda X: [[]] * len(X))
    assert not h.EXP_DIR.exists()


PREPARE = '''
import harness, numpy as np, pandas as pd
rng = np.random.default_rng(25)
parts, groups = [], {}
for split, n in zip(('train','val','test'), (60,18,18)):
    X = rng.normal(scale=.1, size=(n,6))
    if FMT == 'scalar':
        y = 5 + X[:,0]*3 - X[:,1]
    elif FMT == 'binary_score':
        y = np.arange(n) % 2
        X[:,0] += 4*y-2
    else:
        cls = np.arange(n) % 3
        X[np.arange(n),cls] += 5
        y = [['normal','fault_a','fault_b'][i] for i in cls]
    X = pd.DataFrame(X, columns=[f's{i}' for i in range(6)]) if KIND == 'table' else X.astype('float32').reshape(n,2,3)
    parts.append((X,y))
    groups[split] = [f'{split}-machine-{i//6}' for i in range(n)]
harness.write_dataset(*parts, groups=groups, meta={'split':'group','normal_class':'normal'})
'''

TRAIN = '''
import harness
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
X,y = harness.load_train()
steps = [StandardScaler(), LinearRegression() if FMT == 'scalar' else LogisticRegression(max_iter=200)]
if KIND != 'table':
    steps.insert(0, FunctionTransformer(lambda W: W.reshape(len(W),-1)))
model = make_pipeline(*steps).fit(X,y)
harness.save_model(model)
harness.finish(model.predict if FMT == 'scalar' else model.predict_proba, threshold=.5 if FMT == 'binary_score' else None)
'''


@pytest.mark.parametrize("kind", ["table", "tensor"])
@pytest.mark.parametrize("fmt", ["scalar", "binary_score", "class_probs"])
def test_real_prepare_validate_train_save_finish_full_final_package(tmp_path, monkeypatch, kind, fmt):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    ws = workspace.create_workspace("industrial-e2e", _spec(fmt))
    prepare = PREPARE.replace("KIND", repr(kind)).replace("FMT", repr(fmt))
    train = TRAIN.replace("KIND", repr(kind)).replace("FMT", repr(fmt))
    report = runner.run_prepare(ws, prepare)
    assert report.ok and report.checks["prepare_runs"], (report.messages, report.log_tail)
    assert validators.validate(ws).ok
    ws = workspace.freeze(ws)
    before = _snapshot(Path(ws.path) / "data")
    result = runner.run_experiment(ws, train, {"description": f"{kind} {fmt} lifecycle"})
    assert result.status == "keep" and result.all_kpis_passed, result.log_tail
    full = runner.full_train(ws, 1)
    assert full.status == "keep", full.log_tail
    final = runner.final_test(ws)
    assert final.passed, final
    pkg = Path(package.package(ws))
    pred = package.load_predictor(str(pkg))
    assert pred.predict({"replay": 0})["source"] == "replay"
    if kind == "table":
        X = pd.read_parquet(Path(ws.path) / "data" / "test_inputs.parquet").set_index("sample_id")
        live = pred.predict(X.iloc[0].to_dict())
        batch = pred.model.predict(X) if fmt == "scalar" else pred.model.predict_proba(X)
    else:
        X = np.load(Path(ws.path) / "data" / "test_inputs.npy")
        live = pred.predict({"array": X[0].tolist()})
        batch = pred.model.predict(X) if fmt == "scalar" else pred.model.predict_proba(X)
    key = {"scalar": "value", "binary_score": "score", "class_probs": "probs"}[fmt]
    expected = batch[0] if fmt != "binary_score" else batch[0, 1]
    np.testing.assert_allclose(live[key], expected, atol=.001)
    assert live["latency_ms"] >= 0
    assert _snapshot(Path(ws.path) / "data") == before
    manifest = json.loads((pkg / "manifest.json").read_text())
    assert manifest["input_kind"] == ("table" if kind == "table" else "array")
    assert manifest["test_metrics"] == final.test_metrics
