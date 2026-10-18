"""Human-in-the-loop: the agent awaits approval of its task specs; the API resolves it."""

import asyncio
from typing import Any

_pending: dict[str, asyncio.Future] = {}


async def wait_for(run_id: str) -> dict[str, Any]:
    fut = asyncio.get_running_loop().create_future()
    _pending[run_id] = fut
    try:
        return await fut
    finally:
        _pending.pop(run_id, None)


def resolve(run_id: str, decision: dict[str, Any]) -> bool:
    fut = _pending.get(run_id)
    if fut is None or fut.done():
        return False
    fut.get_loop().call_soon_threadsafe(fut.set_result, decision)
    return True


def is_pending(run_id: str) -> bool:
    return run_id in _pending
