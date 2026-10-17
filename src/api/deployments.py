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


def _replays(predictor) -> bool:
    """A real package with held-out test rows: the live feed replays them (a mock predictor has none)."""
    return getattr(predictor, "n_replay", 0) > 0


def _is_camera(predictor) -> bool:
    return _replays(predictor) and getattr(predictor, "input_kind", None) == "files"


def _truth_alarm(predictor, truth) -> bool | None:
    """Should this sample have raised an alarm? Same rule the predictor uses on its own output."""
    if truth is None:
        return None
    if predictor.fmt == "binary_score":
        return bool(truth)
    if predictor.fmt == "scalar":
        return float(truth) <= predictor.horizon
    if predictor.fmt == "class_probs" and predictor.normal_class is not None:
        return truth != predictor.normal_class
    return None


def _replay_tick(dep_id: str, tid: str, predictor, tick: int) -> dict:
    """Next held-out test sample (never seen in training) through the deployed model, for real: an image file,
    or a feature row. Array inputs fall back to the winner's stored prediction for that sample."""
    row = predictor.replay(tick)
    sid = row.pop("sample_id")
    if predictor.input_kind == "files":
        out = predictor.predict({"image_path": row["path"]})
        out["frame_url"] = f"/api/deployments/{dep_id}/frames/{tid}/{sid}"
    elif predictor.input_kind == "table":
        out = predictor.predict(row)
    else:
        out = predictor.predict({"replay": tick})
    truth = predictor.truth(sid)
    out.update(sample_id=sid, truth=truth)
    should = _truth_alarm(predictor, truth)
    if should is not None and "alarm" in out:
        out["outcome"] = {(True, True): "caught", (False, True): "missed",
                          (True, False): "false_alarm", (False, False): "ok"}[(bool(out["alarm"]), should)]
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
    """SSE: every deployed model gets its next held-out test sample (real packages) or a simulated plant
    sample (mock) -> predictions, true label + caught / missed / false_alarm / ok, rolling alarm stats."""
    live = _get(dep_id)
    stream_ids = sorted({s for spec in live.specs.values() for s in spec.input_streams})

    async def stream():
        ticks, alarms = 0, dict.fromkeys(live.predictors, 0)
        async for sample in simulator.live(stream_ids, hz=hz):
            ticks += 1
            preds = {}
            for tid, predictor in live.predictors.items():
                if _replays(predictor):
                    preds[tid] = await asyncio.to_thread(_replay_tick, dep_id, tid, predictor, ticks - 1)
                else:
                    preds[tid] = predictor.predict({**sample["streams"], "risk": sample["risk"]})
                alarms[tid] += bool(preds[tid].get("alarm"))
            yield {"event": "prediction", "data": json.dumps({
                "t": sample["t"], "cycle": sample["cycle"], "predictions": preds,
                "kpi_tracker": {tid: {"alarms": n, "ticks": ticks, "alarm_rate": n / ticks} for tid, n in alarms.items()},
            })}

    return EventSourceResponse(stream())
