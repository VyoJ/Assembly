"""Shared logic for running train.py on remote compute (Daytona now; Modal / any GPU provider later).

A provider subclasses RemoteExecutor and implements five primitives on an opaque `box` (sandbox / container):
    _create_box(ws, label) -> box        _upload(box, data, path)        _download(box, path) -> bytes
    _exec(box, command, timeout_s) -> (rc, output)                       _exec_stream(box, command, timeout_s, on_text) -> rc | None
    _destroy(box)
Everything else lives here: one box per task workspace, syncing the code (src/core + src/ml) and the workspace INPUTS
(harness.py, task.json, program.md, data/ - hidden val/test labels live outside the workspace and are never sent),
running with the same env contract as local runs, and copying run_dir back so scoring stays local.
"""

import hashlib
import io
import shlex
import tarfile
import threading
import time
from importlib import metadata
from pathlib import Path

from core.config import settings
from core.contracts import Workspace
from ml.executors.base import LineSink

SRC = Path(__file__).resolve().parents[2]  # .../src (contains core/ and ml/)
WORKSPACE_INPUTS = ("harness.py", "task.json", "program.md", "data")
MAX_WEIGHTS_MB = 512

# What agent code may import in the box, pinned to the versions installed HERE so the pre-run static check
# (ml.static_check, which inspects local signatures) matches what actually runs remotely.
ML_PACKAGES = [
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "lightgbm", "optuna", "joblib", "onnx",
    "onnxruntime", "pillow", "opencv-python-headless", "pycocotools", "torchmetrics", "timm", "transformers",
    "accelerate", "huggingface-hub", "ultralytics", "matplotlib", "tqdm",
    # imported by src/core + src/ml themselves
    "pydantic", "pydantic-settings", "pyyaml", "python-dotenv",
]
TORCH_PACKAGES = ["torch", "torchvision"]  # installed from a separate index (CPU or CUDA / ROCm build)


def pinned(packages: list[str]) -> list[str]:
    out = []
    for p in packages:
        try:
            out.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            continue
    return out


def requirements_hash(*parts: list[str]) -> str:
    return hashlib.sha256("\n".join(x for part in parts for x in part).encode()).hexdigest()[:10]


def _tar(entries: list[tuple[Path, str]]) -> bytes:
    """tar.gz of (local path, archive name) pairs, skipping caches."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in entries:
            if src.exists():
                tar.add(src, arcname=arc, filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return buf.getvalue()


class RemoteExecutor:
    name = "remote"
    remote_root = "/home/daytona/assembly"  # providers may override after creating a box
    poll_s = 2.0
    cpu_threads: int | None = None  # the box's real CPU allowance (containers often report the host's cores)

    def __init__(self) -> None:
        self._boxes: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()
        self._code_tar: bytes | None = None

    # ------------------------------------------------------------ provider primitives
    def _create_box(self, ws: Workspace, label: str):
        raise NotImplementedError

    def _upload(self, box, data: bytes, path: str) -> None:
        raise NotImplementedError

    def _download(self, box, path: str) -> bytes:
        raise NotImplementedError

    def _exec(self, box, command: str, timeout_s: float = 300) -> tuple[int, str]:
        raise NotImplementedError

    def _exec_stream(self, box, command: str, timeout_s: float, on_text) -> int | None:
        raise NotImplementedError

    def _destroy(self, box) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------ shared logic
    def _ws_dir(self, ws: Workspace) -> str:
        return f"{self.remote_root}/runs/{ws.run_id}/{ws.task_id}"

    def _check(self, box, command: str, what: str) -> str:
        rc, out = self._exec(box, command)
        if rc != 0:
            raise RuntimeError(f"[{self.name}] {what} failed (exit {rc}): {out[-1500:]}")
        return out

    def _put_tar(self, box, data: bytes, dest: str, what: str) -> None:
        tmp = f"/tmp/assembly_{hashlib.sha256(data).hexdigest()[:12]}.tgz"
        self._upload(box, data, tmp)
        self._check(box, f"mkdir -p {shlex.quote(dest)} && tar xzf {tmp} -C {shlex.quote(dest)} && rm -f {tmp}", what)

    def _box(self, ws: Workspace):
        """The task's box, created and synced on first use (one per workspace, reused across experiments)."""
        key = (ws.run_id, ws.task_id)
        with self._lock:
            if key in self._boxes:
                return self._boxes[key]
            box = self._create_box(ws, f"{ws.run_id}-{ws.task_id}")
            try:
                self._sync(box, ws)
            except Exception:
                self._destroy(box)
                raise
            self._boxes[key] = box
            return box

    def _sync(self, box, ws: Workspace) -> None:
        # File inputs (images, recordings) work too: write_dataset copies them into data/files/<split>/ with
        # RELATIVE paths, and harness resolves them against the box's own data dir.
        if self._code_tar is None:
            self._code_tar = _tar([(SRC / "core", "core"), (SRC / "ml", "ml")])
        self._put_tar(box, self._code_tar, f"{self.remote_root}/src", "code sync")
        wsp = Path(ws.path)
        self._put_tar(box, _tar([(wsp / n, n) for n in WORKSPACE_INPUTS]), self._ws_dir(ws), "workspace sync")
        weights = settings.data_dir / "weights"
        if weights.is_dir():
            size_mb = sum(f.stat().st_size for f in weights.rglob("*") if f.is_file()) / 1e6
            if size_mb <= MAX_WEIGHTS_MB:
                self._put_tar(box, _tar([(weights, "weights")]), self.remote_root, "weights sync")

    def _remote_env(self, ws: Workspace, run: str, budget_s: float) -> dict[str, str]:
        wsd = self._ws_dir(ws)
        return {  # same contract as runner._env for train.py: no data dir, no labels dir, no secrets
            "ASSEMBLY_WORKSPACE": wsd,
            "ASSEMBLY_EXP_DIR": f"{wsd}/{run}",
            "ASSEMBLY_TIME_BUDGET_S": str(int(budget_s)),
            "ASSEMBLY_WEIGHTS_DIR": f"{self.remote_root}/weights",
            "PYTHONPATH": f"{wsd}:{self.remote_root}/src",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            # n_jobs=-1 / OpenMP would otherwise start one thread per HOST core and thrash a small box
            **({k: str(self.cpu_threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "LOKY_MAX_CPU_COUNT")} if self.cpu_threads else {}),
        }

    def run(self, ws: Workspace, run_dir: Path, *, env: dict[str, str], budget_s: float, timeout_s: float,
            sink: LineSink) -> tuple[int | None, float]:
        t0 = time.monotonic()
        box = self._box(ws)
        wsd, run = self._ws_dir(ws), run_dir.name
        self._check(box, f"rm -rf {shlex.quote(f'{wsd}/{run}')} && mkdir -p {shlex.quote(f'{wsd}/{run}')}", "run dir")
        self._upload(box, (run_dir / "train.py").read_bytes(), f"{wsd}/{run}/train.py")
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in self._remote_env(ws, run, budget_s).items())
        # `timeout` (coreutils) kills the run inside the box; the provider's own deadline is the backstop
        limit = f"$(command -v timeout >/dev/null && echo timeout -k 15 {int(timeout_s)})"
        cmd = (f"cd {shlex.quote(wsd)} && {limit} env {exports} "
               f"python -u {shlex.quote(f'{run}/train.py')} 2>&1")
        sink.line(f"[{self.name}] running {run}/train.py remotely (timeout {int(timeout_s)}s)")
        rc = self._exec_stream(box, cmd, timeout_s + 60, sink.text)
        if rc in (124, 137):  # killed by `timeout`
            rc = None
        self._fetch(box, ws, run_dir)
        return rc, time.monotonic() - t0

    def _fetch(self, box, ws: Workspace, run_dir: Path) -> None:
        """Copy everything train.py wrote in the remote run dir back into run_dir (even after a crash)."""
        wsd, run = self._ws_dir(ws), run_dir.name
        tmp = f"/tmp/assembly_out_{run}.tgz"
        rc, _ = self._exec(box, f"tar czf {tmp} -C {shlex.quote(wsd)} {shlex.quote(run)}")
        if rc != 0:
            return
        data = self._download(box, tmp)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name != f"{run}/train.py"]  # keep the local snapshot
            tar.extractall(run_dir.parent, members=members, filter="data")
        self._exec(box, f"rm -f {tmp}")

    def release(self, ws: Workspace) -> None:
        with self._lock:
            box = self._boxes.pop((ws.run_id, ws.task_id), None)
        if box is not None:
            self._destroy(box)
