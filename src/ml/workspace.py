"""Task workspaces + experiment bookkeeping (shared by mock and real runs).

runs/{run_id}/{task_id}/
    task.json  program.md  harness.py  prepare.py  data_card.md   (read-only once frozen)
    train.py                                                      (current best; the only editable file)
    data/  best/  exp_000/ ... exp_NNN/  results.tsv  experiments.jsonl
runs/{run_id}/_labels/{task_id}/                                  (hidden val/test labels)
"""

import difflib
import json
import shutil
import stat
from importlib import metadata
from pathlib import Path

from core.config import settings
from core.contracts import ExperimentResult, TaskSpec, Workspace

ML_DIR = Path(__file__).resolve().parent
HARNESS_SRC = ML_DIR / "harness" / "harness.py"
PROGRAM_SRC = ML_DIR / "program.md"
RESULTS_HEADER = "exp\tscore\tprimary\tkpis_passed\tp95_latency_ms\tstatus\tdescription\n"
READ_ONLY_FILES = ["task.json", "program.md", "harness.py", "prepare.py", "data_card.md"]

# Packages agent-written code is likely to use; listed with versions in data_card.md.
KEY_PACKAGES = [
    "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "optuna", "torch", "torchvision",
    "torchmetrics", "timm", "transformers", "accelerate", "ultralytics", "opencv-python-headless", "pillow",
    "onnx", "onnxruntime", "joblib", "pycocotools",
]


def ws_dir(ws: Workspace) -> Path:
    return Path(ws.path)


def labels_dir(run_id: str, task_id: str) -> Path:
    return settings.runs_dir / run_id / "_labels" / task_id


def installed_packages() -> dict[str, str]:
    out = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def create_workspace(run_id: str, spec: TaskSpec) -> Workspace:
    path = settings.runs_dir / run_id / spec.id
    if path.exists():
        _make_writable(path)
        shutil.rmtree(path)
    (path / "data").mkdir(parents=True)
    (path / "best").mkdir()
    labels_dir(run_id, spec.id).mkdir(parents=True, exist_ok=True)
    shutil.copy(HARNESS_SRC, path / "harness.py")
    shutil.copy(PROGRAM_SRC, path / "program.md")
    (path / "task.json").write_text(spec.model_dump_json(indent=2))
    (path / "results.tsv").write_text(RESULTS_HEADER)
    return Workspace(run_id=run_id, task_id=spec.id, path=str(path), spec=spec)


def update_spec(ws: Workspace, spec: TaskSpec) -> Workspace:
    (ws_dir(ws) / "task.json").write_text(spec.model_dump_json(indent=2))
    return ws.model_copy(update={"spec": spec})


def freeze(ws: Workspace) -> Workspace:
    for name in READ_ONLY_FILES:
        p = ws_dir(ws) / name
        if p.exists():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return ws.model_copy(update={"frozen": True})


def _make_writable(path: Path) -> None:
    for p in path.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | stat.S_IWUSR)


def read_file(ws: Workspace, name: str, max_chars: int = 20_000) -> str:
    p = (ws_dir(ws) / name).resolve()
    if not p.is_relative_to(ws_dir(ws).resolve()) or not p.is_file():
        return f"(no such file: {name})"
    text = p.read_text(errors="replace")
    return text if len(text) <= max_chars else text[:max_chars] + "\n...(truncated)"


def write_prepare(ws: Workspace, code: str) -> None:
    (ws_dir(ws) / "prepare.py").write_text(code)


def write_data_card(ws: Workspace, card: str) -> None:
    pkgs = "\n".join(f"- {k}=={v}" for k, v in installed_packages().items()) or "- (ML group not installed)"
    (ws_dir(ws) / "data_card.md").write_text(f"{card.rstrip()}\n\n## Installed packages\n{pkgs}\n")


def best_train_py(ws: Workspace) -> str | None:
    p = ws_dir(ws) / "best" / "train.py"
    return p.read_text() if p.exists() else None


def best_result(ws: Workspace) -> ExperimentResult | None:
    p = ws_dir(ws) / "best" / "result.json"
    return ExperimentResult.model_validate_json(p.read_text()) if p.exists() else None


def next_exp_id(ws: Workspace) -> str:
    n = len(list(ws_dir(ws).glob("exp_*")))
    return f"exp_{n:03d}"


def new_exp_dir(ws: Workspace, train_py: str) -> tuple[str, Path]:
    exp_id = next_exp_id(ws)
    d = ws_dir(ws) / exp_id
    d.mkdir()
    (d / "train.py").write_text(train_py)
    return exp_id, d


def diff_vs_best(ws: Workspace, train_py: str) -> str:
    base = best_train_py(ws) or ""
    return "".join(
        difflib.unified_diff(base.splitlines(True), train_py.splitlines(True), "best/train.py", "train.py")
    )


def record_experiment(ws: Workspace, result: ExperimentResult, train_py: str) -> ExperimentResult:
    """Deterministic keep/discard + bookkeeping. Call after metrics are computed (or the run failed)."""
    d = ws_dir(ws) / result.exp_id
    best = best_result(ws)
    result.diff = result.diff or diff_vs_best(ws, train_py)
    if result.status not in ("crash", "timeout", "invalid_output"):
        improved = result.score is not None and (best is None or best.score is None or result.score < best.score)
        result.status = "keep" if improved else "discard"

    (d / "result.json").write_text(result.model_dump_json(indent=2))
    if result.status == "keep":
        shutil.copy(d / "train.py", ws_dir(ws) / "best" / "train.py")
        shutil.copy(d / "train.py", ws_dir(ws) / "train.py")
        (ws_dir(ws) / "best" / "result.json").write_text(result.model_dump_json(indent=2))
        if (d / "model").exists():
            shutil.rmtree(ws_dir(ws) / "best" / "model", ignore_errors=True)
            shutil.copytree(d / "model", ws_dir(ws) / "best" / "model")

    primary = result.metrics.get(ws.spec.primary_metric)
    passed = sum(result.kpis_passed.values())
    row = [
        result.exp_id,
        f"{result.score:.6f}" if result.score is not None else "-",
        f"{primary:.6f}" if primary is not None else "-",
        f"{passed}/{len(ws.spec.kpis)}",
        f"{result.metrics.get('p95_latency_ms', float('nan')):.1f}",
        result.status,
        result.description.replace("\t", " ").replace("\n", " "),
    ]
    with (ws_dir(ws) / "results.tsv").open("a") as f:
        f.write("\t".join(row) + "\n")
    with (ws_dir(ws) / "experiments.jsonl").open("a") as f:
        f.write(json.dumps(result.model_dump(exclude={"diff", "log_tail"})) + "\n")
    return result


def results_tail(ws: Workspace, n: int) -> str:
    lines = (ws_dir(ws) / "results.tsv").read_text().splitlines()
    return "\n".join([lines[0], *lines[1:][-n:]])


def list_experiments(ws_path: Path) -> list[dict]:
    p = ws_path / "experiments.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
