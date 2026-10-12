"""Fake ML layer (ML_MOCK=1): instant, plausible results so the agent/API/UI can be built
without data or a GPU. Writes real workspace files via ml.workspace, so bookkeeping is real.
"""

import json
import random
import time
from pathlib import Path

from core.config import settings
from core.contracts import (
    ExperimentResult,
    FinalReport,
    StreamInfo,
    StreamProfile,
    TaskSpec,
    ValidationReport,
    Workspace,
)
from ml import catalog, scoring, workspace

_rng = random.Random(7)


def _sleep(mult: float = 1.0) -> None:
    time.sleep(settings.mock_delay_s * mult)


def peek_stream(stream_id: str, n: int = 5) -> str:
    rows = [f"unit=1 cycle={c} value={100 + _rng.gauss(0, 1):.3f}" for c in range(1, n + 1)]
    return f"{stream_id} (mock)\n" + "\n".join(rows)


def profile_streams(spec: TaskSpec, streams: list[StreamInfo]) -> list[StreamProfile]:
    _sleep()
    out = []
    for s in streams:
        hint = catalog.mock_hint(s.id)
        if hint == "constant":
            out.append(StreamProfile(stream_id=s.id, relevance=0.0, keep=False, reason="constant signal (std≈0)"))
        elif hint == "distractor":
            r = round(_rng.uniform(0.0, 0.05), 3)
            out.append(StreamProfile(stream_id=s.id, relevance=r, keep=False, reason=f"no relationship with target (|ρ|={r})"))
        else:
            r = round(_rng.uniform(0.3, 0.9), 3)
            out.append(StreamProfile(stream_id=s.id, relevance=r, keep=True, reason=f"|ρ|={r} with target"))
    return out


def run_prepare(ws: Workspace, prepare_py: str) -> ValidationReport:
    _sleep(2)
    workspace.write_prepare(ws, prepare_py)
    attempt = len(list(Path(ws.path).glob(".prepare_attempt_*")))
    (Path(ws.path) / f".prepare_attempt_{attempt}").touch()
    if attempt == 0 and _rng.random() < 0.4:
        return ValidationReport(
            ok=False,
            checks={"group_disjoint": False, "min_sizes": True, "format": True},
            messages=["group overlap: 12 groups appear in both train and val; split by group, not by row"],
            log_tail="validators.py: group_disjoint FAILED",
        )
    card = (
        f"# Data card: {ws.spec.name} (mock)\n"
        f"- framing: {ws.spec.framing}\n- target: {ws.spec.target_definition}\n"
        f"- prediction_format: {ws.spec.prediction_format}\n- split: {ws.spec.split_strategy}\n"
        f"- inputs: {', '.join(ws.spec.input_streams) or '(none)'}\n"
        "- sizes: train=14,231 val=3,108 test=3,212 (groups: 70/15/15)\n- positive rate: train 9.8%, val 10.1%"
    )
    workspace.write_data_card(ws, card)
    return ValidationReport(
        ok=True, checks={"group_disjoint": True, "min_sizes": True, "format": True}, data_card=card
    )


def _fake_metrics(spec: TaskSpec, progress: float) -> dict[str, float]:
    """Metrics that move from 'misses KPIs' to 'passes KPIs' as progress goes 0 -> 1."""
    out: dict[str, float] = {}
    for k in spec.kpis:
        bad = k.threshold * (1.35 if k.op == "<=" else 0.75)
        good = k.threshold * (0.85 if k.op == "<=" else 1.06)
        v = bad + (good - bad) * progress + _rng.gauss(0, abs(k.threshold) * 0.04)
        m = scoring.get_metric(spec.prediction_format, k.metric)
        out[k.metric] = min(max(v, 0.0), 1.0) if m and m.bounded else max(v, 0.0)
    for name in {spec.primary_metric, *spec.metrics, "p95_latency_ms"} - out.keys():
        m = scoring.get_metric(spec.prediction_format, name)
        if name.endswith("latency_ms"):
            out[name] = round(_rng.uniform(4, 40), 2)
        elif m and m.bounded:
            base = 0.6 + 0.3 * progress
            out[name] = min(max(base + _rng.gauss(0, 0.03), 0.0), 1.0) if m.direction == "max" else max(0.0, 0.3 - 0.25 * progress)
        else:
            out[name] = max(0.0, 40 - 25 * progress + _rng.gauss(0, 1.5))
    return out


def run_experiment(ws: Workspace, train_py: str, meta: dict, progress=None) -> ExperimentResult:
    exp_id, d = workspace.new_exp_dir(ws, train_py)
    n = int(exp_id.split("_")[1])
    for epoch in range(1, 4):
        _sleep()
        if progress:
            progress({"exp_id": exp_id, "epoch": epoch, "loss": round(1.0 / (epoch + n * 0.3), 4)})
    base = {
        "exp_id": exp_id, "task_id": ws.task_id, "description": meta.get("description", ""),
        "hypothesis": meta.get("hypothesis", ""), "train_seconds": round(ws.spec.time_budget_s * _rng.uniform(0.5, 1.0), 1),
    }
    if n > 0 and _rng.random() < 0.12:
        log = "Traceback (most recent call last):\n  File \"train.py\", line 42, in <module>\nRuntimeError: shape mismatch: [64, 24] vs [64, 21]"
        (d / "run.log").write_text(log)
        return workspace.record_experiment(ws, ExperimentResult(status="crash", log_tail=log, **base), train_py)
    metrics = _fake_metrics(ws.spec, min(1.0, n / 6 + _rng.uniform(-0.1, 0.15)))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (d / "run.log").write_text(f"[mock] trained {base['train_seconds']}s\n{json.dumps(metrics)}\n")
    result = ExperimentResult(
        status="discard", score=scoring.compute_score(metrics, ws.spec), metrics=metrics,
        kpis_passed=scoring.kpi_status(metrics, ws.spec.kpis), peak_mem_gb=round(_rng.uniform(2, 24), 1),
        log_tail="(mock run ok)", **base,
    )
    return workspace.record_experiment(ws, result, train_py)


def _nudge_better(spec: TaskSpec, name: str, value: float) -> float:
    """Longer training: a few % better on every metric."""
    m = scoring.get_metric(spec.prediction_format, name)
    if m is None or m.direction == "min":
        return value * 0.97
    return min(1.0, value * 1.02) if m.bounded else value * 1.02


def full_train(ws: Workspace, multiplier: float = 5.0, progress=None) -> ExperimentResult:
    best = workspace.best_result(ws)
    _sleep(3)
    if best is None:
        raise RuntimeError(f"{ws.task_id}: no successful experiment to full-train")
    metrics = {k: _nudge_better(ws.spec, k, v) for k, v in best.metrics.items()}
    return best.model_copy(update={
        "exp_id": "full_train", "metrics": metrics, "score": scoring.compute_score(metrics, ws.spec),
        "kpis_passed": scoring.kpi_status(metrics, ws.spec.kpis),
        "train_seconds": round(ws.spec.time_budget_s * multiplier, 1), "description": f"full train ({multiplier}x budget)",
    })


def final_test(ws: Workspace) -> FinalReport:
    best = workspace.best_result(ws)
    _sleep()
    metrics = {k: v * _rng.uniform(0.98, 1.02) for k, v in (best.metrics if best else {}).items()}
    status = scoring.kpi_status(metrics, ws.spec.kpis)
    return FinalReport(
        task_id=ws.task_id, exp_id=best.exp_id if best else "-", test_metrics=metrics,
        passed=bool(status) and all(status.values()), failed_kpis=[k for k, ok in status.items() if not ok],
    )


def package(ws: Workspace) -> str:
    path = Path(ws.path) / "package"
    path.mkdir(exist_ok=True)
    best = workspace.best_result(ws)
    (path / "manifest.json").write_text(json.dumps({
        "task_id": ws.task_id, "exp_id": best.exp_id if best else None,
        "prediction_format": ws.spec.prediction_format, "mock": True,
    }, indent=2))
    return str(path)


class MockPredictor:
    def __init__(self, manifest: dict):
        self.fmt = manifest["prediction_format"]
        self.threshold = 0.5

    def predict(self, sample: dict) -> dict:
        t0 = time.perf_counter()
        risk = float(sample.get("risk", _rng.random()))
        if self.fmt == "scalar":
            out = {"value": round(max(0.0, 125 * (1 - risk) + _rng.gauss(0, 3)), 1)}
            out["alarm"] = out["value"] < 30
        elif self.fmt == "binary_score":
            s = min(1.0, max(0.0, risk + _rng.gauss(0, 0.05)))
            out = {"score": round(s, 3), "alarm": s > self.threshold}
        elif self.fmt in ("boxes", "masks"):
            defect = _rng.random() < risk
            boxes = [[40, 52, 88, 97, round(_rng.uniform(0.6, 0.95), 2), 0]] if defect else []
            out = {"boxes": boxes, "alarm": bool(boxes)}
        else:
            p = [round(_rng.random(), 3) for _ in range(3)]
            out = {"probs": [x / sum(p) for x in p], "alarm": False}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000 + _rng.uniform(3, 12), 2)
        return out


def load_predictor(package_path: str) -> MockPredictor:
    return MockPredictor(json.loads((Path(package_path) / "manifest.json").read_text()))
