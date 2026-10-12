import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from core import registry
from plant import simulator

router = APIRouter(prefix="/deployments", tags=["deployments"])


def _get(dep_id: str) -> registry.LiveDeployment:
    live = registry.deployments.get(dep_id)
    if live is None:
        raise HTTPException(404, "unknown deployment")
    return live


@router.get("")
async def list_deployments() -> list[dict]:
    return [d.deployment.model_dump() for d in registry.deployments.values()]


@router.post("/{dep_id}/predict")
async def predict(dep_id: str, sample: dict, task_id: str | None = None) -> dict:
    live = _get(dep_id)
    targets = [task_id] if task_id else list(live.predictors)
    return {tid: live.predictors[tid].predict(sample) for tid in targets if tid in live.predictors}


@router.get("/{dep_id}/live")
async def live_predictions(dep_id: str, hz: float = 1.0):
    """SSE: simulated plant samples -> every deployed model -> predictions + rolling alarm stats."""
    live = _get(dep_id)
    stream_ids = sorted({s for spec in live.specs.values() for s in spec.input_streams})

    async def stream():
        ticks, alarms = 0, dict.fromkeys(live.predictors, 0)
        async for sample in simulator.live(stream_ids, hz=hz):
            ticks += 1
            preds = {}
            for tid, predictor in live.predictors.items():
                preds[tid] = predictor.predict({**sample["streams"], "risk": sample["risk"]})
                alarms[tid] += bool(preds[tid].get("alarm"))
            yield {"event": "prediction", "data": json.dumps({
                "t": sample["t"], "cycle": sample["cycle"], "predictions": preds,
                "kpi_tracker": {tid: {"alarms": n, "ticks": ticks, "alarm_rate": n / ticks} for tid, n in alarms.items()},
            })}

    return EventSourceResponse(stream())
