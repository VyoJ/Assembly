"""design (per task: agent writes prepare.py), approve_plan (human), research (per task: the loop)."""

import asyncio

from agent.design.graph import design_graph
from agent.research.graph import run_research
from agent.state import OuterState
from core import approvals
from core.config import settings
from core.contracts import TaskSpec
from core.events import bus, short_reason
from ml import api as ml_api
from ml import scoring

EDITABLE_AT_APPROVAL = ("kpis", "time_budget_s", "primary_metric", "metrics")


async def design(state: OuterState) -> dict:
    run_id = state["run_id"]
    bus.emit(run_id, "stage", stage="design")

    async def one(t: TaskSpec):
        ws = ml_api.create_workspace(run_id, t)
        out = await design_graph.ainvoke({"run_id": run_id, "ws": ws, "attempts": 0})
        report = out.get("report")
        if report is not None and report.ok:
            return t.id, ws, None
        if report is not None:  # keep the reason readable: the full messages are in the validation events
            shown = report.messages[:3]
            more = len(report.messages) - len(shown)
            reason = "; ".join(shown) + (f"; ... and {more} more (see the last validation event)" if more > 0 else "")
        else:
            reason = out.get("error", "no valid prepare.py")
        return t.id, None, f"design failed after {out.get('attempts', 0)} attempts: {reason}"

    results = await asyncio.gather(*(one(t) for t in state["spec"].tasks))
    dropped = {tid: why for tid, _, why in results if why}
    for tid, why in dropped.items():
        bus.emit(run_id, "task_dropped", task_id=tid, reason=short_reason(why))
    workspaces = {tid: ws for tid, ws, _ in results if ws is not None}
    return {"workspaces": workspaces, "dropped": dropped, "status": "running" if workspaces else "no_tasks"}


async def approve_plan(state: OuterState) -> dict:
    run_id, workspaces = state["run_id"], dict(state["workspaces"])
    bus.emit(run_id, "stage", stage="approve_plan")
    if state.get("auto_approve"):
        decision = {"approved": True}
    else:
        bus.emit(run_id, "approval_required", tasks={
            tid: {"spec": ws.spec.model_dump(), "data_card": ml_api.read_workspace_file(ws, "data_card.md")}
            for tid, ws in workspaces.items()
        })
        decision = await approvals.wait_for(run_id)

    if not decision.get("approved", False):
        bus.emit(run_id, "rejected", reason=decision.get("reason", ""))
        return {"status": "rejected"}

    dropped = dict(state.get("dropped", {}))
    for tid in decision.get("drop", []):
        if workspaces.pop(tid, None) is not None:
            dropped[tid] = "dropped by operator"
    edits = {t["id"]: t for t in decision.get("tasks", [])}
    for tid, ws in list(workspaces.items()):
        if tid in edits:
            changed = {k: edits[tid][k] for k in EDITABLE_AT_APPROVAL if k in edits[tid]}
            spec = TaskSpec.model_validate({**ws.spec.model_dump(), **changed})
            spec = spec.model_copy(update={"time_budget_s": settings.clamp_budget(spec.time_budget_s)})
            if not scoring.validate_spec_metrics(spec):
                ws = ml_api.update_spec(ws, spec)
        workspaces[tid] = ml_api.freeze(ws)
    bus.emit(run_id, "approved", tasks=[ws.spec.model_dump() for ws in workspaces.values()])
    return {"workspaces": workspaces, "dropped": dropped, "status": "running" if workspaces else "no_tasks"}


async def research(state: OuterState) -> dict:
    run_id = state["run_id"]
    bus.emit(run_id, "stage", stage="research")
    items = list(state["workspaces"].items())
    outs = await asyncio.gather(*(run_research(run_id, ws) for _, ws in items))
    best, dropped = {}, dict(state.get("dropped", {}))
    for (tid, _), out in zip(items, outs):
        if out.get("best") is not None:
            best[tid] = out["best"]
        else:
            dropped[tid] = f"no successful experiment ({out.get('stop_reason')})"
            bus.emit(run_id, "task_dropped", task_id=tid, reason=short_reason(dropped[tid]))
    return {"best": best, "dropped": dropped}
