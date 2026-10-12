from fastapi import APIRouter, HTTPException

from ml import api as ml_api
from plant import simulator

router = APIRouter(prefix="/streams", tags=["streams"])


@router.get("")
async def list_streams() -> dict:
    streams, labels = ml_api.list_catalog()
    return {"streams": streams, "label_sources": labels}


@router.get("/{stream_id}/preview")
async def preview(stream_id: str, n: int = 50) -> list[dict]:
    try:
        return simulator.preview(stream_id, n)
    except StopIteration:
        raise HTTPException(404, "unknown stream") from None
