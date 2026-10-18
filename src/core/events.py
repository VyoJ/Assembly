"""In-process event bus: agent/ML emit events per run; the API streams them over SSE.

Every event is also appended to runs/{run_id}/events.jsonl. emit() is safe to call from
worker threads (ML calls run in asyncio.to_thread).
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

from fastapi.encoders import jsonable_encoder

from core.config import settings


def short_reason(text: str, keep: int = 3, max_chars: int = 600) -> str:
    """Long '; '-separated message lists (e.g. one validator failure per feature column) -> the first few plus a
    count. Short texts are returned unchanged."""
    if len(text) <= max_chars:
        return text
    parts = text.split("; ")
    if len(parts) <= keep:
        return text[:max_chars] + "..."
    return "; ".join(parts[:keep]) + f"; ... and {len(parts) - keep} more"


class EventBus:
    def __init__(self) -> None:
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, run_id: str, type: str, **data: Any) -> dict:
        event = jsonable_encoder({"type": type, "ts": time.time(), "run_id": run_id, **data})
        self._history[run_id].append(event)
        self._persist(run_id, event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        for q in list(self._subs[run_id]):
            if running is not None:
                q.put_nowait(event)
            elif self._loop is not None:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
        return event

    def subscribe(self, run_id: str, replay: bool = True) -> asyncio.Queue:
        self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        if replay:
            for e in self._history[run_id]:
                q.put_nowait(e)
        self._subs[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        if q in self._subs[run_id]:
            self._subs[run_id].remove(q)

    def load(self, run_id: str, events: list[dict]) -> None:
        """Seed a run's history (e.g. a recorded run read back from events.jsonl) without persisting it again."""
        if not self._history[run_id]:
            self._history[run_id] = list(events)

    def history(self, run_id: str) -> list[dict]:
        return list(self._history[run_id])

    @staticmethod
    def _persist(run_id: str, event: dict) -> None:
        path = settings.runs_dir / run_id / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")


bus = EventBus()
