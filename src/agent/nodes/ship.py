"""full_train (winner x bigger budget), final_gate (hidden test split + report), deploy."""

import asyncio
import uuid

from agent import llm, mock_llm, prompts
from agent.state import OuterState
from core import registry
from core.contracts import Deployment
from core.events import bus
from ml import api as ml_api


def _active(state: OuterState):
    return [(tid, ws) for tid, ws in state["workspaces"].items() if tid in state.get("best", {})]


async def full_train(state: OuterState) -> dict:
    run_id = state["run_id"]
    bus.emit(run_id, "stage", stage="full_train")

    async def one(tid, ws):
        def progress(p: dict) -> None:
            bus.emit(run_id, "progress", task_id=tid, stage="full_train", **p)

        try:
            result = await asyncio.to_thread(ml_api.full_train, ws, None, progress)
        except Exception as e:  # noqa: BLE001 - keep the research winner if full training fails
            bus.emit(run_id, "full_train_failed", task_id=tid, error=repr(e))
            return tid, None
        bus.emit(run_id, "full_train_done", **result.model_dump(exclude={"diff"}))
        return tid, result

    results = await asyncio.gather(*(one(tid, ws) for tid, ws in _active(state)))
    return {"full": {tid: r for tid, r in results if r is not None}}


async def final_gate(state: OuterState) -> dict:
    run_id = state["run_id"]
    bus.emit(run_id, "stage", stage="final_gate")
    reports = {}
    for tid, ws in _active(state):
        report = await asyncio.to_thread(ml_api.final_test, ws)
        reports[tid] = report
        bus.emit(run_id, "final_test", **report.model_dump())
    for ws in state["workspaces"].values():  # no more training after this point: free remote compute
        await asyncio.to_thread(ml_api.release, ws)

    lines = []
    for tid, r in reports.items():
        ws = state["workspaces"][tid]
        metrics = ", ".join(f"{k}={v:.3g}" for k, v in r.test_metrics.items())
        lines.append(f"- {ws.spec.name}: {'PASSED' if r.passed else 'FAILED ' + str(r.failed_kpis)}; {metrics}")
    for tid, why in state.get("dropped", {}).items():
        lines.append(f"- {tid}: not delivered ({why})")
    results = "\n".join(lines)
    summary = await llm.text(
        llm.messages(prompts.REPORT_SYSTEM, prompts.REPORT_USER.format(goal=state["goal"], results=results)),
        mock=lambda: mock_llm.report(state["goal"], results),
    )
    bus.emit(run_id, "report", summary=summary)
    return {"reports": reports, "summary": summary}


async def deploy(state: OuterState) -> dict:
    run_id = state["run_id"]
    bus.emit(run_id, "stage", stage="deploy")
    dep_id = f"dep_{uuid.uuid4().hex[:8]}"
    paths, predictors, specs, exp_ids = {}, {}, {}, []
    for tid, ws in _active(state):
        paths[tid] = await asyncio.to_thread(ml_api.package, ws)
        predictors[tid] = await asyncio.to_thread(ml_api.load_predictor, paths[tid])
        specs[tid] = ws.spec
        exp_ids.append(state["best"][tid].exp_id)
    deployment = Deployment(
        id=dep_id, run_id=run_id, task_ids=list(paths), exp_ids=exp_ids, package_paths=paths,
        endpoint=f"/api/deployments/{dep_id}",
    )
    registry.deployments[dep_id] = registry.LiveDeployment(deployment, specs, predictors)
    bus.emit(run_id, "deployed", deployment=deployment.model_dump())
    return {"deployment": deployment, "status": "done"}
