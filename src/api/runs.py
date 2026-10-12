import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.graph import new_run_id, run_pipeline
from core import approvals
from core.config import settings
from core.events import bus
from ml.workspace import list_experiments

router = APIRouter(prefix="/runs", tags=["runs"])


@dataclass
class RunRecord:
    run_id: str
    goal: str
    status: str = "running"
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    task: asyncio.Task | None = None


RUNS: dict[str, RunRecord] = {}


class StartRun(BaseModel):
    goal: str
    auto_approve: bool = False


class Approval(BaseModel):
    approved: bool = True
    tasks: list[dict] = []  # edited TaskSpecs (only kpis / time_budget_s / primary_metric / metrics apply)
    drop: list[str] = []  # task ids to drop
    reason: str = ""


def _summary(rec: RunRecord) -> dict:
    status = "awaiting_approval" if approvals.is_pending(rec.run_id) else rec.status
    return {"run_id": rec.run_id, "goal": rec.goal, "status": status, "error": rec.error}


async def _execute(rec: RunRecord, auto_approve: bool) -> None:
    def on_state(state: dict) -> None:
        rec.state = jsonable_encoder(state)

    try:
        final = await run_pipeline(rec.goal, run_id=rec.run_id, auto_approve=auto_approve, on_state=on_state)
        rec.status = final.get("status", "done")
    except Exception as e:  # noqa: BLE001 - surfaced via GET /runs/{id}
        rec.status, rec.error = "failed", repr(e)


@router.post("")
async def start_run(body: StartRun) -> dict:
    rec = RunRecord(run_id=new_run_id(), goal=body.goal)
    RUNS[rec.run_id] = rec
    rec.task = asyncio.create_task(_execute(rec, body.auto_approve))
    return _summary(rec)


@router.get("")
async def list_runs() -> list[dict]:
    return [_summary(r) for r in RUNS.values()]


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict:
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(404, "unknown run")
    return {**_summary(rec), "state": rec.state}


@router.get("/{run_id}/events")
async def run_events(run_id: str):
    """SSE: replays past events, then streams new ones. Each event's `event` field is its type."""
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")

    async def stream():
        q = bus.subscribe(run_id)
        try:
            while True:
                e = await q.get()
                yield {"event": e["type"], "data": json.dumps(e)}
        finally:
            bus.unsubscribe(run_id, q)

    return EventSourceResponse(stream())


@router.post("/{run_id}/approve")
async def approve(run_id: str, body: Approval) -> dict:
    if not approvals.resolve(run_id, body.model_dump()):
        raise HTTPException(409, "run is not waiting for approval")
    return {"ok": True}


@router.get("/{run_id}/tasks/{task_id}/experiments")
async def experiments(run_id: str, task_id: str) -> list[dict]:
    return list_experiments(settings.runs_dir / run_id / task_id)
