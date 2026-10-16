"""Where agent-written train.py runs. The runner calls one method; executors differ only in where the process lives.

    Executor.run(ws, run_dir, env=..., budget_s=..., timeout_s=..., sink=...) -> (returncode | None on timeout, seconds)

Contract: when run() returns, run_dir (exp_NNN/ or full/) holds everything train.py wrote there (model/,
predictions, timings.json, ...) on THIS machine, so scoring, keep/discard and packaging stay local and unchanged.
Hidden labels never leave this machine. Executors: local (subprocess), daytona (CPU sandboxes); a GPU provider
(e.g. Modal) plugs in by subclassing RemoteExecutor and implementing its five primitives.
"""

import re
import time
from collections import deque
from pathlib import Path
from typing import Protocol

from core.contracts import Workspace

LOG_TAIL_LINES = 40
PROGRESS_MIN_INTERVAL_S = 0.5
_KV = re.compile(r"([A-Za-z_][\w./-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_progress(line: str) -> dict:
    """'epoch=3 loss=0.124 val_rmse: 17.2' -> {"epoch": 3, "loss": 0.124, "val_rmse": 17.2}"""
    out = {}
    for k, v in _KV.findall(line):
        num = float(v)
        out[k] = int(num) if num.is_integer() and "." not in v and "e" not in v.lower() else num
    return out


class LineSink:
    """Collects a run's output wherever it runs: writes run.log, keeps the tail, forwards throttled progress."""

    def __init__(self, log_path: Path, progress=None, tag: dict | None = None):
        self.log = log_path.open("w")
        self.tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self.progress, self.tag, self._last = progress, tag or {}, 0.0

    def line(self, line: str) -> None:
        line = line.rstrip("\n")
        self.log.write(line + "\n")
        self.log.flush()
        self.tail.append(line)
        if self.progress and time.monotonic() - self._last >= PROGRESS_MIN_INTERVAL_S:
            self._last = time.monotonic()
            try:
                self.progress({**self.tag, "line": line[-300:], **parse_progress(line)})
            except Exception as e:  # noqa: BLE001 - a UI callback must never kill a run
                self.log.write(f"[runner] progress callback failed: {e!r}\n")

    def text(self, chunk: str) -> None:
        for ln in chunk.splitlines():
            self.line(ln)

    def tail_text(self) -> str:
        return "\n".join(self.tail)

    def close(self) -> None:
        self.log.close()


class Executor(Protocol):
    name: str

    def run(self, ws: Workspace, run_dir: Path, *, env: dict[str, str], budget_s: float, timeout_s: float,
            sink: LineSink) -> tuple[int | None, float]:
        """Run run_dir/train.py for this workspace; return (returncode or None on timeout, wall seconds)."""
        ...

    def release(self, ws: Workspace) -> None:
        """Free any compute held for this workspace (no-op locally)."""
        ...


_instances: dict[str, Executor] = {}


def get_executor(name: str) -> Executor:
    if name not in _instances:
        if name == "local":
            from ml.executors.local import LocalExecutor

            _instances[name] = LocalExecutor()
        elif name == "daytona":
            from ml.executors.daytona import DaytonaExecutor

            _instances[name] = DaytonaExecutor()
        else:
            raise ValueError(f"unknown EXECUTOR '{name}' (local | daytona)")
    return _instances[name]
