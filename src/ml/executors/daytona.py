"""Daytona CPU sandboxes (free credits don't cover GPUs; a GPU provider is another RemoteExecutor subclass).

One snapshot holds the pinned ML stack (CPU torch); it is built once per requirements hash and reused. Each task
gets its own sandbox from that snapshot, with outbound network blocked (experiments must not download anything)
and auto-stop / auto-delete as a safety net if this process dies without releasing it.
"""

import time
import uuid

from core.config import settings
from core.contracts import Workspace
from ml.executors.remote import ML_PACKAGES, TORCH_PACKAGES, RemoteExecutor, pinned, requirements_hash

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


# System libraries the Python stack needs on Linux: ultralytics pulls the full OpenCV, which needs these.
APT_PACKAGES = ["libgl1", "libglib2.0-0", "libxcb1"]


def snapshot_name() -> str:
    return f"assembly-ml-cpu-{requirements_hash(APT_PACKAGES, pinned(ML_PACKAGES), pinned(TORCH_PACKAGES))}"


class DaytonaExecutor(RemoteExecutor):
    name = "daytona"

    def __init__(self) -> None:
        super().__init__()
        from daytona import Daytona, DaytonaConfig

        if not settings.daytona_api_key:
            raise RuntimeError("EXECUTOR=daytona needs DAYTONA_API_KEY in .env")
        self.client = Daytona(DaytonaConfig(api_key=settings.daytona_api_key))
        self._snapshot: str | None = None
        self.cpu_threads = settings.daytona_cpu

    # ------------------------------------------------------------ snapshot (built once)
    def ensure_snapshot(self, on_logs=print) -> str:
        if self._snapshot:
            return self._snapshot
        from daytona import CreateSnapshotParams, Image, Resources

        name = settings.daytona_snapshot or snapshot_name()
        try:
            snap = self.client.snapshot.get(name)
            if str(getattr(snap, "state", "")).lower().endswith("inactive"):
                self.client.snapshot.activate(snap)
        except Exception:  # noqa: BLE001 - not found: build it
            torch = pinned(TORCH_PACKAGES)
            image = (Image.debian_slim("3.12")
                     .run_commands("apt-get update && apt-get install -y --no-install-recommends "
                                   + " ".join(APT_PACKAGES) + " && rm -rf /var/lib/apt/lists/*")
                     .pip_install(*torch, index_url=TORCH_CPU_INDEX)
                     .pip_install(*pinned(ML_PACKAGES))
                     .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}))
            self.client.snapshot.create(
                CreateSnapshotParams(name=name, image=image, resources=Resources(
                    cpu=settings.daytona_cpu, memory=settings.daytona_memory_gb, disk=settings.daytona_disk_gb)),
                on_logs=on_logs,
            )
        self._snapshot = name
        return name

    # ------------------------------------------------------------ primitives
    def _create_box(self, ws: Workspace, label: str):
        from daytona import CreateSandboxFromSnapshotParams

        sandbox = self.client.create(CreateSandboxFromSnapshotParams(
            snapshot=self.ensure_snapshot(),
            labels={"app": "assembly", "run": ws.run_id},  # one box per run, shared by its tasks
            auto_stop_interval=settings.daytona_auto_stop_min,
            auto_delete_interval=settings.daytona_auto_delete_min,
            network_block_all=True,
        ), timeout=300)
        self.remote_root = f"{sandbox.get_user_home_dir().rstrip('/')}/assembly"
        return sandbox

    def _is_capacity_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return "limit exceeded" in text or "quota" in text or "no available runners" in text

    def _upload(self, box, data: bytes, path: str) -> None:
        box.fs.upload_file(data, path)

    def _download(self, box, path: str) -> bytes:
        return box.fs.download_file(path)

    def _exec(self, box, command: str, timeout_s: float = 300) -> tuple[int, str]:
        r = box.process.exec(f"bash -lc {_q(command)}", timeout=int(timeout_s))
        return int(r.exit_code or 0), r.result or ""

    def _exec_stream(self, box, command: str, timeout_s: float, on_text) -> int | None:
        """Async session command; poll its output and exit code. Past timeout_s the session is deleted."""
        from daytona import SessionExecuteRequest

        sid = f"run-{uuid.uuid4().hex[:8]}"
        box.process.create_session(sid)
        try:
            cmd = box.process.execute_session_command(sid, SessionExecuteRequest(
                command=f"bash -lc {_q(command)}", run_async=True))
            seen, deadline = 0, time.monotonic() + timeout_s
            while True:
                time.sleep(self.poll_s)
                out = box.process.get_session_command_logs(sid, cmd.cmd_id).output or ""
                if len(out) > seen:
                    on_text(out[seen:])
                    seen = len(out)
                code = box.process.get_session_command(sid, cmd.cmd_id).exit_code
                if code is not None:
                    out = box.process.get_session_command_logs(sid, cmd.cmd_id).output or ""
                    if len(out) > seen:
                        on_text(out[seen:])
                    return int(code)
                if time.monotonic() > deadline:
                    return None
        finally:
            try:
                box.process.delete_session(sid)  # also stops anything still running in it
            except Exception:  # noqa: BLE001, S110 - the session may already be gone
                pass

    def _destroy(self, box) -> None:
        try:
            box.delete()
        except Exception:  # noqa: BLE001, S110 - auto-delete is the safety net
            pass


def _q(s: str) -> str:
    import shlex

    return shlex.quote(s)
