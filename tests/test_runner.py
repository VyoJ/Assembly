"""Real runner + scoring + packaging on a tiny synthetic task (no harness needed: the scripts write the
ML_PLAN.md file layout directly). Skipped unless the `ml` dependency group is installed."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from core.contracts import KPI, TaskSpec
from ml import package, runner, scoring, validators, workspace

PREPARE = """
import json, os
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
n = 600
X = pd.DataFrame({"s1": rng.normal(size=n), "s2": rng.normal(size=n)})
y = 60 + 25 * X["s1"] - 10 * X["s2"] + rng.normal(scale=3, size=n)
X.insert(0, "sample_id", [f"u{i}" for i in range(n)])
labels = pd.DataFrame({"sample_id": X["sample_id"], "y": y})
data, hidden = Path("data"), Path(os.environ["ASSEMBLY_LABELS_DIR"])
parts = {"train": slice(0, 400), "val": slice(400, 500), "test": slice(500, 600)}
X[parts["train"]].to_parquet(data / "train_inputs.parquet", index=False)
labels[parts["train"]].to_parquet(data / "train_labels.parquet", index=False)
for s in ("val", "test"):
    X[parts[s]].to_parquet(data / f"{s}_inputs.parquet", index=False)
    labels[parts[s]].to_parquet(hidden / f"{s}_labels.parquet", index=False)
(data / "meta.json").write_text(json.dumps({"prediction_format": "scalar", "early_warning_horizon": 40}))
print("prepared", n)
"""

TRAIN = """
import json, os
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from ml.models.export import save_model

exp = Path(os.environ["ASSEMBLY_EXP_DIR"])
X = pd.read_parquet("data/train_inputs.parquet").set_index("sample_id")
y = pd.read_parquet("data/train_labels.parquet").set_index("sample_id")["y"]
model = {MODEL}.fit(X, y)
print("epoch=1 loss=0.5")
save_model(model, str(exp / "model"), example_input=X.head(4))
for split in ("val", "test"):
    Xs = pd.read_parquet(f"data/{split}_inputs.parquet").set_index("sample_id")
    pd.DataFrame({"sample_id": Xs.index, "pred": model.predict(Xs)}).to_parquet(exp / f"{split}_predictions.parquet", index=False)
(exp / "timings.json").write_text(json.dumps({"p50_latency_ms": 0.1, "p95_latency_ms": 0.2}))
"""


@pytest.fixture
def ws(monkeypatch):
    monkeypatch.setattr(
        validators, "validate",
        lambda ws: scoring_ok() if (Path(ws.path) / "data" / "meta.json").exists() else None,
    )
    spec = TaskSpec(
        id="toy_rul", name="toy", rationale="", framing="", input_streams=[], label_sources=[],
        target_definition="", prediction_format="scalar", primary_metric="rmse",
        metrics=["rmse", "mae", "early_warning_recall", "p95_latency_ms"],
        kpis=[KPI(metric="rmse", op="<=", threshold=6.0)], split_strategy="rows", time_budget_s=30,
    )
    return workspace.create_workspace("test-runner", spec)


def scoring_ok():
    from core.contracts import ValidationReport

    return ValidationReport(ok=True, checks={"files_present": True}, data_card="# toy data card")


def test_prepare_experiments_full_train_final_and_package(ws, monkeypatch):
    report = runner.run_prepare(ws, PREPARE)
    assert report.ok and report.checks["prepare_runs"]
    assert (Path(ws.path) / "data_card.md").exists()
    assert not (Path(ws.path) / "data" / "val_labels.parquet").exists()  # labels stay hidden

    events = []
    weak = runner.run_experiment(ws, TRAIN.replace("{MODEL}", "LinearRegression(fit_intercept=False)"),
                                 {"description": "no intercept"}, events.append)
    assert weak.status == "keep" and not weak.all_kpis_passed
    assert any(e.get("epoch") == 1 and e.get("loss") == 0.5 for e in events)

    good = runner.run_experiment(ws, TRAIN.replace("{MODEL}", "LinearRegression()"), {"description": "linreg"})
    assert good.status == "keep" and good.all_kpis_passed and good.score < weak.score
    assert good.metrics["p95_latency_ms"] == 0.2 and "early_warning_recall" in good.metrics

    crash = runner.run_experiment(ws, "raise SystemExit('boom')", {"description": "crash"})
    assert crash.status == "crash" and "boom" in crash.log_tail

    invalid = runner.run_experiment(ws, "print('forgot to predict')", {"description": "no preds"})
    assert invalid.status == "invalid_output" and "harness.finish" in invalid.log_tail

    monkeypatch.setattr(runner, "KILL_GRACE_S", 0)
    monkeypatch.setattr(ws.spec, "time_budget_s", 1)
    slow = runner.run_experiment(ws, "import time; time.sleep(30)", {"description": "too slow"})
    assert slow.status == "timeout"

    rows = (Path(ws.path) / "results.tsv").read_text().splitlines()
    assert [r.split("\t")[5] for r in rows[1:]] == ["keep", "keep", "crash", "invalid_output", "timeout"]

    monkeypatch.setattr(ws.spec, "time_budget_s", 30)
    full = runner.full_train(ws, 2)
    assert full.status == "keep"  # same code + seed -> same score, so the full-train copy is used
    assert runner.winner_dir(ws).name == "full"

    final = runner.final_test(ws)
    assert final.passed and final.test_metrics["rmse"] < 6

    pkg = package.package(ws)
    manifest = json.loads((Path(pkg) / "manifest.json").read_text())
    assert manifest["feature_columns"] == ["s1", "s2"] and manifest["early_warning_horizon"] == 40
    pred = package.load_predictor(pkg)
    out = pred.predict({"s1": 1.0, "s2": 0.0})
    assert abs(out["value"] - 85) < 3 and out["alarm"] is False and "latency_ms" in out
    assert "value" in pred.predict({"replay": 3})


def test_compute_metrics_formats(tmp_path):
    import pandas as pd

    ids = [f"s{i}" for i in range(6)]
    pd.DataFrame({"sample_id": ids, "y": [1, 0, 1, 0, 1, 0]}).to_parquet(tmp_path / "labels.parquet")
    pd.DataFrame({"sample_id": ids[::-1], "score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3][::-1]}).to_parquet(tmp_path / "p.parquet")
    m = scoring.compute_metrics("binary_score", str(tmp_path / "p"), str(tmp_path / "labels"),
                                ["auroc", "recall", "false_alarm_rate"], {"threshold": 0.5})
    assert m == {"auroc": 1.0, "recall": 1.0, "false_alarm_rate": 0.0}  # matched by sample_id, not row order

    pd.DataFrame({"sample_id": ids[:3], "score": [0.9, 0.1, 0.8]}).to_parquet(tmp_path / "short.parquet")
    with pytest.raises(scoring.PredictionError, match="no prediction for 3"):
        scoring.compute_metrics("binary_score", str(tmp_path / "short"), str(tmp_path / "labels"), ["auroc"])

    gt = {"a": [[0, 0, 10, 10, 0]], "b": []}
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "pb.json").write_text(json.dumps({"a": [[0, 0, 10, 10, 0.9, 0]], "b": [[1, 1, 3, 3, 0.2, 0]]}))
    m = scoring.compute_metrics("boxes", str(tmp_path / "pb"), str(tmp_path / "gt"),
                                ["map50", "image_recall", "image_far"])
    assert m == {"map50": 1.0, "image_recall": 1.0, "image_far": 0.0}

    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    pd.DataFrame({"sample_id": ids[:3], "y": [0, 1, 1]}).to_parquet(tmp_path / "cl.parquet")
    pd.DataFrame({"sample_id": ids[:3], "p_0": probs[:, 0], "p_1": probs[:, 1]}).to_parquet(tmp_path / "pc.parquet")
    m = scoring.compute_metrics("class_probs", str(tmp_path / "pc"), str(tmp_path / "cl"), ["accuracy", "min_class_recall"])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["min_class_recall"] == 0.5


# ---------------------------------------------------------------- the real thing: harness + validators, no stubs

REAL_PREPARE = """
import harness, numpy as np, pandas as pd
rng = np.random.default_rng(1)
rows = []
for unit in range(40):
    life = int(rng.integers(120, 220))
    for t in range(1, life + 1):
        wear = t / life
        rows.append({"unit": unit, "time": t, "s1": 5 * wear + rng.normal(0, .2), "s2": rng.normal(), "rul": life - t})
df = pd.DataFrame(rows)
units = rng.permutation(40)
split = {"train": units[:28], "val": units[28:34], "test": units[34:]}
parts, groups = {}, {}
for name, us in split.items():
    part = df[df.unit.isin(us)]
    parts[name] = (part[["time", "s1", "s2"]], part["rul"].clip(upper=125))
    groups[name] = part["unit"].tolist()
harness.write_dataset(parts["train"], parts["val"], parts["test"], groups=groups,
                      meta={"split": "group", "group_col": "unit", "early_warning_horizon": 30})
"""

REAL_TRAIN = """
import harness
from sklearn.ensemble import HistGradientBoostingRegressor
X, y = harness.load_train()
model = HistGradientBoostingRegressor(max_iter=200).fit(X, y)
print("epoch=1 loss=1.0")
harness.save_model(model)
harness.finish(model.predict)
"""


def test_real_harness_validators_runner_package():
    spec = TaskSpec(
        id="real_rul", name="real", rationale="", framing="", input_streams=[], label_sources=[],
        target_definition="", prediction_format="scalar", primary_metric="rmse",
        metrics=["rmse", "nasa_score", "early_warning_recall", "early_warning_far", "p95_latency_ms"],
        kpis=[KPI(metric="rmse", op="<=", threshold=40.0)], split_strategy="by unit", time_budget_s=60,
    )
    ws = workspace.create_workspace("test-real", spec)
    report = runner.run_prepare(ws, REAL_PREPARE)
    assert report.ok, report.messages
    assert "Installed packages" in (Path(ws.path) / "data_card.md").read_text()

    result = runner.run_experiment(ws, REAL_TRAIN, {"description": "hgb baseline"})
    assert result.status == "keep", result.log_tail
    assert result.all_kpis_passed and result.metrics["p95_latency_ms"] > 0

    runner.full_train(ws, 1.5)
    assert runner.final_test(ws).passed
    pred = package.load_predictor(package.package(ws))
    replayed = pred.predict({"replay": 0})
    assert replayed["source"] == "replay" and "value" in replayed
    live = pred.predict({"time": 150, "s1": 4.9, "s2": 0.0})
    assert live["value"] < 60 and "latency_ms" in live


def test_real_validators_catch_group_leak():
    spec = TaskSpec(
        id="leaky", name="leaky", rationale="", framing="", input_streams=[], label_sources=[],
        target_definition="", prediction_format="scalar", primary_metric="rmse", metrics=["rmse"],
        kpis=[KPI(metric="rmse", op="<=", threshold=40.0)], split_strategy="by unit", time_budget_s=60,
    )
    ws = workspace.create_workspace("test-real", spec)
    leaky = REAL_PREPARE.replace('"val": units[28:34]', '"val": units[20:34]')  # units 20-27 in train AND val
    report = runner.run_prepare(ws, leaky)
    assert not report.ok and not report.checks["group_disjoint"]
    assert any("overlap" in m for m in report.messages)


# ---------------------------------------------------------------- non-tabular inputs: waveforms (array) and images (files)

ARRAY_PREPARE = """
import harness, numpy as np
rng = np.random.default_rng(2)
def make(n, start):
    y = rng.integers(0, 3, n)                       # 3 fault classes
    t = np.linspace(0, 1, 256)
    W = np.stack([np.sin(2 * np.pi * (5 + 10 * c) * t) + rng.normal(0, .3, 256) for c in y])
    return W.astype("float32"), ["normal", "inner_race", "ball"][0:3] and [["normal", "inner_race", "ball"][c] for c in y], \\
        [f"rec{start + i // 10}" for i in range(n)]
(Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make(300, 0), make(90, 100), make(90, 200)
harness.write_dataset((Xtr, ytr), (Xva, yva), (Xte, yte), groups={"train": gtr, "val": gva, "test": gte},
                      meta={"split": "group"})
"""

ARRAY_TRAIN = """
import harness, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
spectrum = FunctionTransformer(lambda W: np.abs(np.fft.rfft(W, axis=1))[:, :64])
X, y = harness.load_train()
model = make_pipeline(spectrum, LogisticRegression(max_iter=500)).fit(X, y.to_numpy())
harness.save_model(model)
harness.finish(model.predict_proba)
"""

FILES_PREPARE = """
import harness, numpy as np
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(3)
out = Path("raw_images"); out.mkdir(exist_ok=True)
def make(n, start):
    paths, ys = [], []
    for i in range(n):
        defect = int(rng.random() < 0.4)
        img = (rng.random((32, 32)) * 60).astype("uint8")
        if defect:
            img[8:20, 8:20] = 255                     # bright scratch
        p = out / f"part{start + i}.png"; Image.fromarray(img).save(p)
        paths.append(str(p)); ys.append(defect)
    return paths, ys
tr, va, te = make(120, 0), make(40, 1000), make(40, 2000)
harness.write_dataset(tr, va, te, meta={"split": "random"})
"""

FILES_TRAIN = """
import harness, numpy as np
from sklearn.linear_model import LogisticRegression
def feats(X):
    return np.stack([[harness.load_image(p).astype(float).max(), harness.load_image(p).astype(float).mean()] for p in X["path"]])
X, y = harness.load_train()
clf = LogisticRegression().fit(feats(X), y.to_numpy())
harness.save_model(clf, example_input=feats(X.head(4)))
harness.finish(lambda X: clf.predict_proba(feats(X))[:, 1], threshold=0.5)
"""


def _spec(task_id: str, fmt: str, primary: str, kpi: KPI) -> TaskSpec:
    return TaskSpec(
        id=task_id, name=task_id, rationale="", framing="", input_streams=[], label_sources=[],
        target_definition="", prediction_format=fmt, primary_metric=primary, metrics=[primary, "p95_latency_ms"],
        kpis=[kpi], split_strategy="", time_budget_s=60,
    )


def test_waveform_array_inputs_class_probs_end_to_end():
    ws = workspace.create_workspace("test-array", _spec("fault", "class_probs", "macro_f1",
                                                        KPI(metric="accuracy", op=">=", threshold=0.9)))
    report = runner.run_prepare(ws, ARRAY_PREPARE)
    assert report.ok, report.messages
    result = runner.run_experiment(ws, ARRAY_TRAIN, {"description": "fft + logreg"})
    assert result.status == "keep" and result.all_kpis_passed, result.log_tail
    assert runner.final_test(ws).passed
    pred = package.load_predictor(package.package(ws))
    out = pred.predict({"replay": 1})
    assert out["label"] in ("normal", "inner_race", "ball") and len(out["probs"]) == 3


def test_image_file_inputs_binary_end_to_end():
    ws = workspace.create_workspace("test-files", _spec("defect", "binary_score", "auroc",
                                                        KPI(metric="recall", op=">=", threshold=0.9)))
    report = runner.run_prepare(ws, FILES_PREPARE)
    assert report.ok, report.messages
    result = runner.run_experiment(ws, FILES_TRAIN, {"description": "brightness logreg"})
    assert result.status == "keep" and result.all_kpis_passed, result.log_tail
    assert runner.final_test(ws).passed
    pred = package.load_predictor(package.package(ws))
    assert pred.predict({"replay": 0})["source"] == "replay"


# split="random" is only accepted for standalone images; windows, video frames and duplicates must fail.
RANDOM_ARRAYS = """
import harness, numpy as np
rng = np.random.default_rng(0)
def make(n):
    return rng.normal(size=(n, 64)).astype("float32"), list(rng.integers(0, 2, n))
harness.write_dataset(make(60), make(20), make(20), meta={"split": "random"})
"""


@pytest.mark.parametrize("variant", ["arrays", "video_frames", "duplicate_image"])
def test_random_split_rejected_where_samples_are_not_independent(variant):
    spec = _spec(f"rand-{variant}", "binary_score", "auroc", KPI(metric="recall", op=">=", threshold=0.9))
    ws = workspace.create_workspace(f"rand-{variant}", spec)
    prepare = {
        "arrays": RANDOM_ARRAYS,
        "video_frames": FILES_PREPARE.replace('Path("raw_images")', 'Path("_cache/frames/cam/clip0")')
                                     .replace("out.mkdir(exist_ok=True)", "out.mkdir(parents=True, exist_ok=True)"),
        # the first test image is a byte-for-byte copy of the first train image
        "duplicate_image": FILES_PREPARE.replace("tr, va, te = make(120, 0), make(40, 1000), make(40, 2000)",
                                                 "tr, va, te = make(120, 0), make(40, 1000), make(40, 2000)\n"
                                                 "te[0][0] = tr[0][0]; te[1][0] = tr[1][0]"),
    }[variant]
    report = runner.run_prepare(ws, prepare)
    assert not report.ok and not report.checks["group_disjoint"], report.messages


def test_random_split_of_standalone_images_passes_with_a_warning():
    spec = _spec("rand-ok", "binary_score", "auroc", KPI(metric="recall", op=">=", threshold=0.9))
    report = runner.run_prepare(workspace.create_workspace("rand-ok", spec), FILES_PREPARE)
    assert report.ok, report.messages
    assert any(m.startswith("WARN: random split") for m in report.messages)
