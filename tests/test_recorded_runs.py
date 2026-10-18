"""Pipeline runs from earlier processes are listed and replayed from runs/{id}/events.jsonl."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api import runs
from api.main import app
from core.config import settings
from core.events import bus


def _write(root, run_id, events):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("".join(json.dumps({"run_id": run_id, "ts": i, **e}) + "\n"
                                            for i, e in enumerate(events)))


def test_recorded_runs_are_listed_and_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    monkeypatch.setattr(runs, "RUNS", {})
    _write(tmp_path, "20261018-100000-aaaa", [
        {"type": "run_started", "goal": "stop stoppages"}, {"type": "stage", "stage": "formulate"},
        {"type": "run_finished", "status": "done"}])
    _write(tmp_path, "20261018-110000-bbbb", [  # killed mid-run; last line cut off
        {"type": "run_started", "goal": "killed run"}, {"type": "stage", "stage": "design"}])
    (tmp_path / "20261018-110000-bbbb" / "events.jsonl").open("a").write('{"type": "tool_ca')
    _write(tmp_path, "20261018-120000-cccc", [{"type": "exploration", "task_id": "t"}])  # run_research.py session

    client = TestClient(app)
    listed = {r["run_id"]: r for r in client.get("/api/runs").json()}
    assert set(listed) == {"20261018-100000-aaaa", "20261018-110000-bbbb"}
    assert listed["20261018-100000-aaaa"] == {"run_id": "20261018-100000-aaaa", "goal": "stop stoppages",
                                                   "status": "done", "error": None, "source": "recorded"}
    killed = client.get("/api/runs/20261018-110000-bbbb").json()
    assert killed["status"] == "failed" and killed["error"].startswith("interrupted")
    assert [e["type"] for e in bus.history("20261018-110000-bbbb")] == ["run_started", "stage", "run_failed"]
    assert client.get("/api/runs/20261018-120000-cccc").status_code == 404
    # the file is never rewritten
    assert '"run_failed"' not in (tmp_path / "20261018-110000-bbbb" / "events.jsonl").read_text()


def test_recorded_deployment_is_listed_and_activated_on_first_use(tmp_path, monkeypatch):
    from core import registry
    from core.contracts import KPI, TaskSpec

    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    monkeypatch.setattr(registry, "deployments", {})
    spec = TaskSpec(id="t", name="t", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="binary_score", primary_metric="auroc",
                    metrics=["auroc"], kpis=[KPI(metric="auroc", op=">=", threshold=0.5)],
                    split_strategy="by unit", time_budget_s=60)
    dep = {"id": "dep_old", "run_id": "20261018-100000-aaaa", "task_ids": ["t"], "exp_ids": ["exp_000"],
           "package_paths": {"t": str(tmp_path / "nowhere")}, "endpoint": "/api/deployments/dep_old", "status": "live"}
    _write(tmp_path, "20261018-100000-aaaa", [{"type": "run_started", "goal": "g"}, {"type": "deployed", "deployment": dep},
                                              {"type": "run_finished", "status": "done"}])
    (tmp_path / "nowhere").mkdir()
    (tmp_path / "nowhere" / "manifest.json").write_text(json.dumps({"task_id": "t", "prediction_format": "binary_score",
                                                                    "mock": True}))
    (tmp_path / "20261018-100000-aaaa" / "t").mkdir()
    (tmp_path / "20261018-100000-aaaa" / "t" / "task.json").write_text(spec.model_dump_json())

    client = TestClient(app)
    assert [(d["id"], d["status"]) for d in client.get("/api/deployments").json()] == [("dep_old", "stopped")]
    out = client.post("/api/deployments/dep_old/predict", json={"risk": 0.9}).json()  # a mock run's package
    assert "t" in out
    assert [(d["id"], d["status"]) for d in client.get("/api/deployments").json()] == [("dep_old", "live")]
    assert client.post("/api/deployments/dep_nope/predict", json={}).status_code == 404


def test_short_reason_keeps_the_first_messages():
    from core.events import short_reason

    long = "design failed: " + "; ".join(f"FAIL format: feature 'f{i}' must be finite" for i in range(500))
    out = short_reason(long)
    assert out.endswith("... and 497 more") and out.count("FAIL") == 3
    assert short_reason("short") == "short"


def test_relocate_rebases_paths_recorded_on_another_machine(tmp_path, monkeypatch):
    from ml.workspace import relocate

    monkeypatch.setattr(settings, "runs_dir", tmp_path / "vol" / "runs")
    target = tmp_path / "vol" / "runs" / "r1" / "t" / "package"
    target.mkdir(parents=True)
    assert relocate("/Users/someone/Assembly/runs/r1/t/package") == target
    assert relocate("/Users/someone/Assembly/runs/r2/t/package") == Path("/Users/someone/Assembly/runs/r2/t/package")
    assert relocate(target) == target


def test_only_one_run_at_a_time(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    monkeypatch.setattr(runs, "RUNS", {})

    async def scenario():
        blocker = asyncio.Event()
        rec = runs.RunRecord(run_id="20261018-170000-live", goal="g")
        rec.task = asyncio.create_task(blocker.wait())
        runs.RUNS[rec.run_id] = rec
        from fastapi import HTTPException

        try:
            await runs.start_run(runs.StartRun(goal="another"))
            raise AssertionError("second run was accepted")
        except HTTPException as e:
            assert e.status_code == 409 and "20261018-170000-live" in e.detail
        blocker.set()
        await rec.task
        assert runs._active_run() is None

    asyncio.run(scenario())
