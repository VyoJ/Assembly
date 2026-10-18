"""Local subprocess execution (also used for prepare.py and exploration snippets, which always run locally)."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.contracts import Workspace
from ml.executors.base import LineSink


def run_process(cmd: list[str], cwd: Path, env: dict, timeout_s: float, sink: LineSink) -> tuple[int | None, float]:
    """Run cmd, stream stdout+stderr into sink. -> (returncode | None on timeout, seconds)."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,  # own process group, so the kill also takes dataloader workers etc.
    )

    def pump() -> None:
        for line in proc.stdout:
            sink.line(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill(proc)
        rc = None
    reader.join(timeout=5)
    return rc, time.monotonic() - t0


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


class LocalExecutor:
    name = "local"

    def run(self, ws: Workspace, run_dir: Path, *, env: dict[str, str], budget_s: float, timeout_s: float,
            sink: LineSink) -> tuple[int | None, float]:
        return run_process([sys.executable, str(run_dir / "train.py")], Path(ws.path), env, timeout_s, sink)

    def release(self, ws: Workspace) -> None:
        pass
