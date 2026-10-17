"""Live feed of a real deployment: every tick is the next held-out test row through the deployed model, with
its true label and whether the alarm was right (sensor tables here; cameras in test_vision.py)."""

import json

import pytest

pytest.importorskip("sklearn")

from test_runner import REAL_PREPARE, REAL_TRAIN

from core import registry
from core.config import settings
from core.contracts import KPI, Deployment, TaskSpec
from ml import package, runner, workspace


async def test_live_sensor_feed_replays_held_out_rows(tmp_path, monkeypatch):
    from api import deployments

    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    spec = TaskSpec(id="rul", name="rul", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="scalar", primary_metric="rmse",
                    metrics=["rmse", "p95_latency_ms"], kpis=[KPI(metric="rmse", op="<=", threshold=40.0)],
                    split_strategy="by unit", time_budget_s=60)
    ws = workspace.create_workspace("live", spec)
    assert runner.run_prepare(ws, REAL_PREPARE).ok
    assert runner.run_experiment(ws, REAL_TRAIN, {"description": "hgb"}).status == "keep"
    runner.final_test(ws)
    pred = package.load_predictor(package.package(ws))

    dep = Deployment(id="dep_rul", run_id="r", task_ids=["rul"], exp_ids=["exp_000"], package_paths={},
                     endpoint="/api/deployments/dep_rul")
    registry.deployments["dep_rul"] = registry.LiveDeployment(dep, {"rul": spec}, {"rul": pred})
    try:
        response = await deployments.live_predictions("dep_rul", hz=50)
        ticks = []
        async for event in response.body_iterator:
            ticks.append(json.loads(event["data"])["predictions"]["rul"])
            if len(ticks) == 5:
                break
    finally:
        registry.deployments.pop("dep_rul", None)

    assert [t["sample_id"] for t in ticks] == [pred.replay(i)["sample_id"] for i in range(5)]
    stored = [pred.predict({"replay": i})["value"] for i in range(5)]
    assert [t["value"] for t in ticks] == pytest.approx(stored, abs=1e-2)  # the model really ran on the row
    assert len({t["value"] for t in ticks}) > 1  # not a constant
    for t in ticks:
        assert isinstance(t["truth"], (int, float)) and t["outcome"] in ("caught", "missed", "false_alarm", "ok")
