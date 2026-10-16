"""Runs agent-written code (prepare.py / train.py) in a subprocess and scores the result.

Bookkeeping (snapshots, keep/discard, results.tsv) lives in ml.workspace.record_experiment; the runner
executes, times, and scores. File layout: ML_PLAN.md.

Subprocess contract (env vars passed to the child; harness.py reads them):
    ASSEMBLY_WORKSPACE      workspace dir (also the cwd, and on PYTHONPATH so `import harness` works)
    ASSEMBLY_EXP_DIR        exp_NNN/ (or full/) dir to write model/, predictions, timings into
    ASSEMBLY_TIME_BUDGET_S  training budget (spec.time_budget_s, x multiplier for full_train)
    ASSEMBLY_WEIGHTS_DIR    settings.data_dir / "weights"
    ASSEMBLY_DATA_DIR       settings.data_dir (prepare.py only)
    ASSEMBLY_LABELS_DIR     hidden labels dir (prepare.py ONLY; never passed to train.py)
LLM_* and anything that looks like a secret are stripped. train.py is hard-killed at 2x the budget -> "timeout".
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from core.config import settings
from core.contracts import ExperimentResult, FinalReport, ValidationReport, Workspace
from ml import data_overview, scoring, validators, workspace

PREPARE_TIMEOUT_S = 900
KILL_FACTOR = 2.0  # hard kill at KILL_FACTOR x budget (+ grace for loading data and writing predictions)
KILL_GRACE_S = 60
LOG_TAIL_LINES = 40
PROGRESS_MIN_INTERVAL_S = 0.5
_SECRET = re.compile(r"(^LLM_|^ASSEMBLY_|KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_KV = re.compile(r"([A-Za-z_][\w./-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


# ---------------------------------------------------------------- subprocess plumbing

def _env(ws: Workspace, exp_dir: Path | None, budget_s: float, *, prepare: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET.search(k)}
    wsp = str(Path(ws.path).resolve())
    env.update({
        "ASSEMBLY_WORKSPACE": wsp,
        "ASSEMBLY_TIME_BUDGET_S": str(int(budget_s)),
        "ASSEMBLY_WEIGHTS_DIR": str((settings.data_dir / "weights").resolve()),
        "PYTHONPATH": wsp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if exp_dir is not None:
        env["ASSEMBLY_EXP_DIR"] = str(exp_dir.resolve())
    if prepare:
        env["ASSEMBLY_DATA_DIR"] = str(settings.data_dir.resolve())
        env["ASSEMBLY_LABELS_DIR"] = str(workspace.labels_dir(ws.run_id, ws.task_id).resolve())
    return env


def _parse_progress(line: str) -> dict:
    """'epoch=3 loss=0.124 val_rmse: 17.2' -> {"epoch": 3, "loss": 0.124, "val_rmse": 17.2}"""
    out = {}
    for k, v in _KV.findall(line):
        num = float(v)
        out[k] = int(num) if num.is_integer() and "." not in v and "e" not in v.lower() else num
    return out


def _run(cmd: list[str], cwd: Path, env: dict, timeout_s: float, log_path: Path, progress=None, tag: dict | None = None):
    """Run cmd, tee stdout+stderr to log_path, forward throttled progress. -> (returncode | None on timeout, tail, secs)."""
    tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
    t0 = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,  # own process group, so the kill also takes dataloader workers etc.
        )

        def pump() -> None:
            last_emit = 0.0
            for line in proc.stdout:
                log.write(line)
                tail.append(line.rstrip("\n"))
                if progress and time.monotonic() - last_emit >= PROGRESS_MIN_INTERVAL_S:
                    last_emit = time.monotonic()
                    try:
                        progress({**(tag or {}), "line": line.rstrip()[-300:], **_parse_progress(line)})
                    except Exception as e:  # noqa: BLE001 - a UI callback must never kill a run
                        log.write(f"[runner] progress callback failed: {e!r}\n")

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill(proc)
            rc = None
        reader.join(timeout=5)
    return rc, "\n".join(tail), time.monotonic() - t0


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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- scoring one run directory

def _metric_names(ws: Workspace) -> list[str]:
    spec = ws.spec
    return sorted({spec.primary_metric, *spec.metrics, *(k.metric for k in spec.kpis), "p50_latency_ms", "p95_latency_ms"})


def _score_dir(ws: Workspace, run_dir: Path, split: str) -> dict[str, float]:
    """Metrics for run_dir/{split}_predictions vs the hidden {split} labels. Raises scoring.PredictionError."""
    timings = _read_json(run_dir / "timings.json")
    params = {
        **_read_json(Path(ws.path) / "data" / "meta.json"),
        **_read_json(run_dir / "predict_meta.json"),
        "p50_latency_ms": timings.get("p50_latency_ms"),
        "p95_latency_ms": timings.get("p95_latency_ms"),
    }
    return scoring.compute_metrics(
        ws.spec.prediction_format,
        str(run_dir / f"{split}_predictions"),
        str(workspace.labels_dir(ws.run_id, ws.task_id) / f"{split}_labels"),
        _metric_names(ws),
        params,
    )


def _execute(ws: Workspace, run_dir: Path, budget_s: float, base: dict, progress=None) -> ExperimentResult:
    """Run run_dir/train.py, then score it on val. Status is crash / timeout / invalid_output / discard
    (record_experiment turns 'discard' into 'keep' when it beats the best)."""
    env = _env(ws, run_dir, budget_s, prepare=False)
    timeout = budget_s * KILL_FACTOR + KILL_GRACE_S
    rc, tail, secs = _run(
        [sys.executable, str(run_dir / "train.py")], Path(ws.path), env, timeout, run_dir / "run.log",
        progress, {"exp_id": base["exp_id"]},
    )
    base = {**base, "train_seconds": round(secs, 1), "log_tail": tail}
    if rc is None:
        return ExperimentResult(status="timeout", **{**base, "log_tail": f"{tail}\n[runner] killed after {timeout:.0f}s "
                                                     f"(budget {budget_s:.0f}s); size training with harness.time_left()"})
    if rc != 0:
        return ExperimentResult(status="crash", **base)
    try:
        metrics = _score_dir(ws, run_dir, "val")
        if scoring.find_file(run_dir / "test_predictions") is None:
            raise scoring.PredictionError("no test_predictions file; harness.finish() must predict val AND test")
    except scoring.PredictionError as e:
        return ExperimentResult(status="invalid_output", **{**base, "log_tail": f"{tail}\n[runner] invalid output: {e}"})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    peak = _read_json(run_dir / "timings.json").get("peak_mem_gb")
    return ExperimentResult(
        status="discard", score=scoring.compute_score(metrics, ws.spec), metrics=metrics,
        kpis_passed=scoring.kpi_status(metrics, ws.spec.kpis),
        peak_mem_gb=float(peak) if peak is not None else None, **base,
    )


# ---------------------------------------------------------------- public (called via ml.api)

def run_prepare(ws: Workspace, prepare_py: str) -> ValidationReport:
    """Write + run prepare.py (fresh data/ and hidden labels dir), then validate its output."""
    workspace.write_prepare(ws, prepare_py)
    wsp = Path(ws.path)
    labels = workspace.labels_dir(ws.run_id, ws.task_id)
    for d in (wsp / "data", labels):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    rc, tail, _ = _run(
        [sys.executable, "prepare.py"], wsp, _env(ws, None, PREPARE_TIMEOUT_S, prepare=True),
        PREPARE_TIMEOUT_S, wsp / "prepare.log",
    )
    if rc != 0:
        why = f"prepare.py timed out after {PREPARE_TIMEOUT_S}s" if rc is None else f"prepare.py exited with code {rc}"
        return ValidationReport(ok=False, checks={"prepare_runs": False}, messages=[why], log_tail=tail)

    report = validators.validate(ws)
    report.checks = {"prepare_runs": True, **report.checks}
    report.log_tail = report.log_tail or tail
    if report.ok:
        report.data_card = f"{report.data_card.rstrip()}\n\n{data_overview.overview(wsp)}".rstrip()
        workspace.write_data_card(ws, report.data_card)
    return report


SNIPPET_TIMEOUT_S = 60
SNIPPET_MAX_CHARS = 4000


def run_snippet(ws: Workspace, code: str, *, phase: str = "research") -> str:
    """Run a short exploration script and return its (capped) output. Same sandbox rules as train.py: never the
    hidden labels dir. phase="design" also passes ASSEMBLY_DATA_DIR so harness.read_stream / read_labels work
    before prepare.py exists; it still can't write a dataset (no labels dir)."""
    wsp = Path(ws.path)
    scratch = wsp / "_scratch"
    scratch.mkdir(exist_ok=True)
    n = len(list(scratch.glob("snippet_*.py")))
    path = scratch / f"snippet_{n:03d}.py"
    path.write_text(code)
    env = _env(ws, None, SNIPPET_TIMEOUT_S, prepare=False)
    if phase == "design":
        env["ASSEMBLY_DATA_DIR"] = str(settings.data_dir.resolve())
    log = scratch / f"snippet_{n:03d}.log"
    rc, _, secs = _run([sys.executable, str(path)], wsp, env, SNIPPET_TIMEOUT_S, log)
    out = log.read_text(errors="replace")
    if len(out) > SNIPPET_MAX_CHARS:
        half = SNIPPET_MAX_CHARS // 2
        out = f"{out[:half]}\n...({len(out) - SNIPPET_MAX_CHARS} chars cut)...\n{out[-half:]}"
    status = "timed out" if rc is None else f"exit code {rc}"
    return f"[{status}, {secs:.1f}s]\n{out or '(no output; print what you want to see)'}"


def run_experiment(ws: Workspace, train_py: str, meta: dict, progress=None) -> ExperimentResult:
    """Snapshot to exp_NNN/, run under spec.time_budget_s, score vs hidden val labels, keep/discard."""
    exp_id, d = workspace.new_exp_dir(ws, train_py)
    base = {
        "exp_id": exp_id, "task_id": ws.task_id,
        "description": meta.get("description", ""), "hypothesis": meta.get("hypothesis", ""),
    }
    result = _execute(ws, d, ws.spec.time_budget_s, base, progress)
    return workspace.record_experiment(ws, result, train_py)


def full_train(ws: Workspace, multiplier: float, progress=None) -> ExperimentResult:
    """Re-run best/train.py with budget x multiplier into full/; score on val (not recorded in results.tsv).
    status 'keep' if it beats the research winner on val (then it's what final_test and package use)."""
    best, code = workspace.best_result(ws), workspace.best_train_py(ws)
    if best is None or code is None:
        raise RuntimeError(f"{ws.task_id}: no successful experiment to full-train")
    d = Path(ws.path) / "full"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    (d / "train.py").write_text(code)
    base = {
        "exp_id": "full_train", "task_id": ws.task_id, "hypothesis": best.hypothesis,
        "description": f"full train of {best.exp_id} ({multiplier:g}x budget)",
    }
    result = _execute(ws, d, ws.spec.time_budget_s * multiplier, base, progress)
    if result.status == "discard" and result.score is not None and best.score is not None and result.score <= best.score:
        result.status = "keep"
    (d / "result.json").write_text(result.model_dump_json(indent=2))
    return result


def winner_dir(ws: Workspace) -> Path:
    """The run dir final_test and package use: full/ if full-train beat the research winner on val, else its exp dir."""
    wsp = Path(ws.path)
    full = _read_json(wsp / "full" / "result.json")
    if full.get("status") == "keep":
        return wsp / "full"
    best = workspace.best_result(ws)
    if best is None:
        raise RuntimeError(f"{ws.task_id}: no successful experiment")
    return wsp / best.exp_id


def final_test(ws: Workspace) -> FinalReport:
    """Score the winner's TEST predictions vs the hidden test labels (the only time test labels are read)."""
    d = winner_dir(ws)
    exp_id = "full_train" if d.name == "full" else d.name
    try:
        metrics = _score_dir(ws, d, "test")
    except scoring.PredictionError as e:
        return FinalReport(task_id=ws.task_id, exp_id=exp_id, test_metrics={}, passed=False,
                           failed_kpis=[f"invalid test predictions: {e}"])
    status = scoring.kpi_status(metrics, ws.spec.kpis)
    report = FinalReport(
        task_id=ws.task_id, exp_id=exp_id, test_metrics=metrics,
        passed=bool(status) and all(status.values()), failed_kpis=[k for k, ok in status.items() if not ok],
    )
    (Path(ws.path) / "final_report.json").write_text(report.model_dump_json(indent=2))
    return report
