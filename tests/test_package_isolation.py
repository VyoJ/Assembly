"""IsolatedPredictor: the deployed model in its own worker process (default on macOS) answers exactly like the
in-process Predictor, through the same attributes and methods the API uses."""

import pytest

pytest.importorskip("sklearn")

from test_runner import REAL_PREPARE, REAL_TRAIN

from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import package, runner, workspace


def test_isolated_predictor_matches_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    spec = TaskSpec(id="rul", name="rul", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="scalar", primary_metric="rmse",
                    metrics=["rmse", "p95_latency_ms"], kpis=[KPI(metric="rmse", op="<=", threshold=40.0)],
                    split_strategy="by unit", time_budget_s=60)
    ws = workspace.create_workspace("iso", spec)
    assert runner.run_prepare(ws, REAL_PREPARE).ok
    assert runner.run_experiment(ws, REAL_TRAIN, {"description": "hgb"}).status == "keep"
    runner.final_test(ws)
    pkg = package.package(ws)

    local = package.load_predictor(pkg, isolate=False)
    iso = package.load_predictor(pkg, isolate=True)
    try:
        assert isinstance(iso, package.IsolatedPredictor)
        for attr in ("fmt", "input_kind", "horizon", "normal_class", "n_replay"):
            assert getattr(iso, attr) == getattr(local, attr), attr
        row = iso.replay(0)
        assert row == local.replay(0)
        sid = row["sample_id"]
        assert iso.truth(sid) == local.truth(sid)
        sample = {k: v for k, v in row.items() if k != "sample_id"}
        a, b = iso.predict(sample), local.predict(sample)
        assert a["value"] == b["value"] and a["alarm"] == b["alarm"]
        with pytest.raises(RuntimeError, match="AttributeError"):  # worker errors come back; the worker survives
            iso._call("no_such_method")
        assert iso.predict(sample)["value"] == b["value"]
    finally:
        iso.close()
