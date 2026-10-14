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
