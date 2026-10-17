import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
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


def _is_camera(predictor) -> bool:
    """Image tasks with a real package: the live feed replays its held-out test images (a mock has none)."""
    return getattr(predictor, "input_kind", None) == "files" and getattr(predictor, "n_replay", 0) > 0


def _camera_frame(dep_id: str, tid: str, predictor, tick: int) -> dict:
    """Next held-out test image (never seen in training) through the deployed model, for real."""
    frame = predictor.replay(tick)
    out = predictor.predict({"image_path": frame["path"]})
    sid = frame["sample_id"]
    truth = predictor.truth(sid)
    out.update(sample_id=sid, frame_url=f"/api/deployments/{dep_id}/frames/{tid}/{sid}", truth=truth)
    if truth is not None and predictor.fmt == "binary_score":
        out["outcome"] = {(True, True): "caught", (False, True): "missed",
                          (True, False): "false_alarm", (False, False): "ok"}[(out["alarm"], bool(truth))]
    return out


@router.get("/{dep_id}/frames/{task_id}/{sample_id}")
async def frame(dep_id: str, task_id: str, sample_id: str) -> FileResponse:
    """The image behind a live camera prediction (only images in the package's replay set)."""
    predictor = _get(dep_id).predictors.get(task_id)
    if predictor is None or not _is_camera(predictor):
        raise HTTPException(404, "no camera feed for this task")
    paths = predictor.replay_paths()
    if sample_id not in paths or not Path(paths[sample_id]).is_file():
        raise HTTPException(404, "unknown frame")
    return FileResponse(paths[sample_id])


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
                if _is_camera(predictor):
                    preds[tid] = await asyncio.to_thread(_camera_frame, dep_id, tid, predictor, ticks - 1)
                else:
                    preds[tid] = predictor.predict({**sample["streams"], "risk": sample["risk"]})
                alarms[tid] += bool(preds[tid].get("alarm"))
            yield {"event": "prediction", "data": json.dumps({
                "t": sample["t"], "cycle": sample["cycle"], "predictions": preds,
                "kpi_tracker": {tid: {"alarms": n, "ticks": ticks, "alarm_rate": n / ticks} for tid, n in alarms.items()},
            })}

    return EventSourceResponse(stream())
