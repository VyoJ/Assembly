"""RemoteExecutor's shared logic (sync, env contract, run, fetch, release) against a fake provider whose 'remote
machine' is a temp dir. A real provider (Daytona, Modal) only implements the same five primitives."""

import json
import subprocess
from pathlib import Path

import pytest

from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import runner, workspace
from ml.executors import base
from ml.executors.remote import RemoteExecutor

TRAIN = """
import json, os
import harness
X, y = harness.load_train()
print("rows", len(X), "budget", os.environ["ASSEMBLY_TIME_BUDGET_S"])
assert "ASSEMBLY_LABELS_DIR" not in os.environ and "ASSEMBLY_DATA_DIR" not in os.environ
mean = float(y.mean())
harness.save_model(None) if False else None
harness.finish(lambda X: [mean] * len(X))
"""

PREPARE = """
import harness, numpy as np, pandas as pd
rng = np.random.default_rng(0)
def part(n, units):
    X = pd.DataFrame({"a": rng.normal(size=n)}); y = pd.Series(rng.normal(size=n))
    return (X, y), [units[i % len(units)] for i in range(n)]
tr, gtr = part(40, [1, 2, 3, 4]); va, gva = part(20, [5, 6]); te, gte = part(20, [7, 8])
harness.write_dataset(tr, va, te, groups={"train": gtr, "val": gva, "test": gte}, meta={"split": "group"})
"""


class DirExecutor(RemoteExecutor):
    """Fake provider: the 'box' is a directory; commands run with bash locally."""

    name = "fake"

    def __init__(self, root: Path):
        super().__init__()
        self.root, self.created, self.destroyed = root, [], []
        self.poll_s = 0.01

    def _create_box(self, ws, label):
        self.remote_root = str(self.root / label / "assembly")
        self.created.append(label)
        return label

    def _upload(self, box, data, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    def _download(self, box, path):
        return Path(path).read_bytes()

    def _exec(self, box, command, timeout_s=300):
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout_s, check=False)
        return r.returncode, r.stdout + r.stderr

    def _exec_stream(self, box, command, timeout_s, on_text):
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout_s, check=False)
        on_text(r.stdout + r.stderr)
        return r.returncode

    def _destroy(self, box):
        self.destroyed.append(box)


def _make_ws(task_id: str):
    spec = TaskSpec(id=task_id, name=task_id, rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="scalar", primary_metric="rmse", metrics=["rmse"],
                    kpis=[KPI(metric="rmse", op="<=", threshold=10.0)], split_strategy="by unit", time_budget_s=60)
    w = workspace.create_workspace("r1", spec)
    assert runner.run_prepare(w, PREPARE).ok
    return workspace.freeze(w)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    return _make_ws("t")


def test_remote_run_syncs_runs_and_fetches_outputs(ws, tmp_path, monkeypatch):
    fake = DirExecutor(tmp_path / "remote")
    monkeypatch.setattr(settings, "executor", "fake")
    monkeypatch.setitem(base._instances, "fake", fake)

    result = runner.run_experiment(ws, TRAIN, {"description": "remote mean"})
    assert result.status == "keep", result.log_tail
    exp = Path(ws.path) / "exp_000"
    assert (exp / "val_predictions.parquet").exists() and (exp / "timings.json").exists()  # fetched back
    assert "[fake] running exp_000/train.py remotely" in (exp / "run.log").read_text()
    assert "rows 40" in (exp / "run.log").read_text()

    # hidden labels never reach the remote side; the box is reused for the next experiment
    remote_ws = Path(fake.remote_root) / "runs" / "r1" / "t"
    assert (remote_ws / "data" / "train_labels.parquet").exists()
    sent = list(Path(fake.remote_root).rglob("*"))
    assert not any("_labels" in p.parts or p.name in ("val_labels.parquet", "test_labels.parquet") for p in sent)
    runner.run_experiment(ws, TRAIN.replace("mean]", "mean + 0.5]"), {"description": "again"})
    assert fake.created == ["r1"]

    runner.release(ws)
    assert fake.destroyed == ["r1"]


def test_executor_failure_is_reported_as_crash(ws, monkeypatch):
    class Broken(DirExecutor):
        def _create_box(self, ws, label):
            raise RuntimeError("no capacity")

    monkeypatch.setattr(settings, "executor", "broken")
    monkeypatch.setitem(base._instances, "broken", Broken(Path("/nonexistent")))
    result = runner.run_experiment(ws, TRAIN, {"description": "x"})
    assert result.status == "crash" and "no capacity" in result.log_tail
    assert json.loads((Path(ws.path) / "exp_000" / "result.json").read_text())["status"] == "crash"


def test_tasks_of_a_run_share_one_box(ws, tmp_path, monkeypatch):
    fake = DirExecutor(tmp_path / "remote")
    monkeypatch.setattr(settings, "executor", "fake")
    monkeypatch.setitem(base._instances, "fake", fake)
    other = _make_ws("u")
    synced = []
    orig = fake._sync_workspace
    monkeypatch.setattr(fake, "_sync_workspace", lambda box, w: synced.append(w.task_id) or orig(box, w))

    assert runner.run_experiment(ws, TRAIN, {"description": "t0"}).status == "keep"
    assert runner.run_experiment(other, TRAIN, {"description": "u0"}).status == "keep"  # same exp_000 name
    assert runner.run_experiment(ws, TRAIN.replace("mean]", "mean + 1]"), {"description": "t1"}).status in ("keep", "discard")
    assert fake.created == ["r1"] and synced == ["t", "u"]  # one box; each workspace synced once
    assert (Path(other.path) / "exp_000" / "val_predictions.parquet").exists()

    runner.release(ws)
    assert fake.destroyed == []  # 'u' still uses the box
    runner.release(other)
    assert fake.destroyed == ["r1"]


def test_no_capacity_falls_back_to_local(ws, monkeypatch):
    class Full(DirExecutor):
        def _create_box(self, ws, label):
            raise RuntimeError("Failed to create sandbox: Total memory limit exceeded. Maximum allowed: 10GiB.")

        def _is_capacity_error(self, error):
            return "limit exceeded" in str(error)

    monkeypatch.setattr(settings, "executor", "full")
    monkeypatch.setitem(base._instances, "full", Full(Path("/nonexistent")))
    result = runner.run_experiment(ws, TRAIN, {"description": "x"})
    assert result.status == "keep", result.log_tail
    log = (Path(ws.path) / "exp_000" / "run.log").read_text()
    assert "no capacity" in log and "running exp_000 locally" in log

