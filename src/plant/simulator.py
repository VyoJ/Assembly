"""Simulated plant: yields live samples for deployed models.

Currently synthetic (a press that degrades, fails, and gets repaired; random camera parts).
Camera streams of deployed image tasks replay their held-out test images instead (api/deployments.py).
TODO: replay real C-MAPSS test units.
"""

import asyncio
import math
import random
import time
from collections.abc import AsyncIterator

from core.contracts import StreamInfo
from ml import catalog


class Press:
    def __init__(self, life: int = 200, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.life = life
        self.cycle = 0

    def step(self) -> dict:
        self.cycle += 1
        if self.cycle > self.life:  # failure -> maintenance -> new unit
            self.cycle = 1
            self.life = self.rng.randint(150, 300)
        risk = self.cycle / self.life
        return {"cycle": self.cycle, "risk": round(risk, 3)}


def sample_streams(streams: list[StreamInfo], risk: float, rng: random.Random) -> dict:
    values = {}
    for i, s in enumerate(streams):
        if s.modality == "image":
            values[s.id] = {"frame_id": rng.randint(0, 10**6)}
        else:
            drift = 0.0 if catalog.mock_hint(s.id) else risk * (5 + i % 7)
            values[s.id] = round(100 + 10 * math.sin(i) + drift + rng.gauss(0, 0.5), 3)
    return values


async def live(stream_ids: list[str] | None = None, hz: float = 1.0, seed: int = 0) -> AsyncIterator[dict]:
    streams, _ = catalog.list_catalog()
    if stream_ids:
        streams = [s for s in streams if s.id in stream_ids]
    press, rng = Press(seed=seed), random.Random(seed + 1)
    while True:
        state = press.step()
        yield {"t": time.time(), **state, "streams": sample_streams(streams, state["risk"], rng)}
        await asyncio.sleep(1.0 / hz)


def preview(stream_id: str, n: int = 50) -> list[dict]:
    streams, _ = catalog.list_catalog()
    s = next(s for s in streams if s.id == stream_id)
    press, rng = Press(seed=3), random.Random(4)
    out = []
    for _ in range(n):
        state = press.step()
        out.append({"cycle": state["cycle"], "value": sample_streams([s], state["risk"], rng)[s.id]})
    return out
