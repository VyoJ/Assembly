"""Fixed evaluation core: prediction formats, metric registry, KPI checks, scalar score.

The agent may only choose metrics/KPIs that exist here. Score is LOWER = BETTER and
meeting all KPIs always beats missing any.

TODO(ML): implement `compute_metrics` per format (predictions vs. hidden labels).
"""

from dataclasses import dataclass
from typing import Literal

from core.contracts import KPI, PredictionFormat, TaskSpec


@dataclass(frozen=True)
class Metric:
    direction: Literal["min", "max"]
    bounded: bool  # True if the metric lives in [0, 1]
    description: str


_LATENCY = {
    "p50_latency_ms": Metric("min", False, "median inference latency per sample on the target device"),
    "p95_latency_ms": Metric("min", False, "95th percentile inference latency per sample"),
}

METRIC_REGISTRY: dict[PredictionFormat, dict[str, Metric]] = {
    "scalar": {
        "rmse": Metric("min", False, "root mean squared error"),
        "mae": Metric("min", False, "mean absolute error"),
        "r2": Metric("max", False, "coefficient of determination"),
        "nasa_score": Metric("min", False, "asymmetric RUL score penalising late predictions (C-MAPSS)"),
        "early_warning_recall": Metric("max", True, "recall of 'failure within N' alarms derived from the scalar"),
        "early_warning_far": Metric("min", True, "false alarm rate of 'failure within N' alarms"),
    },
    "binary_score": {
        "auroc": Metric("max", True, "area under ROC curve"),
        "pr_auc": Metric("max", True, "area under precision-recall curve"),
        "recall": Metric("max", True, "recall at the submitted threshold"),
        "precision": Metric("max", True, "precision at the submitted threshold"),
        "f1": Metric("max", True, "F1 at the submitted threshold"),
        "false_alarm_rate": Metric("min", True, "FP / (FP + TN) at the submitted threshold"),
    },
    "class_probs": {
        "accuracy": Metric("max", True, "top-1 accuracy"),
        "macro_f1": Metric("max", True, "macro-averaged F1"),
        "min_class_recall": Metric("max", True, "worst per-class recall"),
    },
    "boxes": {
        "map50": Metric("max", True, "mAP at IoU 0.5"),
        "map50_95": Metric("max", True, "COCO mAP at IoU 0.5:0.95"),
        "image_recall": Metric("max", True, "fraction of defective images with >=1 detection above threshold"),
        "image_far": Metric("min", True, "fraction of good images with >=1 detection above threshold"),
    },
    "masks": {
        "mask_map": Metric("max", True, "mask mAP"),
        "iou": Metric("max", True, "mean IoU of defect masks"),
        "image_recall": Metric("max", True, "fraction of defective images flagged"),
        "image_far": Metric("min", True, "fraction of good images flagged"),
    },
}
for _metrics in METRIC_REGISTRY.values():
    _metrics.update(_LATENCY)

KPI_PENALTY = 1000.0  # per failed KPI; dominates everything else
VIOLATION_WEIGHT = 10.0  # times the relative size of each violation


def registry_for_prompt() -> dict[str, dict[str, str]]:
    return {
        fmt: {name: f"{m.direction}: {m.description}" for name, m in metrics.items()}
        for fmt, metrics in METRIC_REGISTRY.items()
    }


def get_metric(fmt: PredictionFormat, name: str) -> Metric | None:
    return METRIC_REGISTRY.get(fmt, {}).get(name)


def kpi_key(k: KPI) -> str:
    return f"{k.metric} {k.op} {k.threshold:g}"


def kpi_passed(value: float | None, k: KPI) -> bool:
    if value is None:
        return False
    return value <= k.threshold if k.op == "<=" else value >= k.threshold


def kpi_status(metrics: dict[str, float], kpis: list[KPI]) -> dict[str, bool]:
    return {kpi_key(k): kpi_passed(metrics.get(k.metric), k) for k in kpis}


def _relative_violation(value: float | None, k: KPI) -> float:
    if value is None:
        return 1.0
    gap = (value - k.threshold) if k.op == "<=" else (k.threshold - value)
    return max(0.0, gap) / max(abs(k.threshold), 1e-6)


def _normalized_primary(metrics: dict[str, float], spec: TaskSpec) -> float:
    m = get_metric(spec.prediction_format, spec.primary_metric)
    value = metrics.get(spec.primary_metric)
    if m is None or value is None:
        return 1.0
    if m.bounded:
        return value if m.direction == "min" else 1.0 - value
    ref = next((k.threshold for k in spec.kpis if k.metric == spec.primary_metric), None)
    if ref:
        return value / abs(ref) if m.direction == "min" else abs(ref) / max(value, 1e-6)
    return value if m.direction == "min" else -value


def compute_score(metrics: dict[str, float], spec: TaskSpec) -> float:
    """Lower is better. Failed KPIs first, then how far off they are, then the primary metric."""
    failed = [k for k in spec.kpis if not kpi_passed(metrics.get(k.metric), k)]
    violation = sum(_relative_violation(metrics.get(k.metric), k) for k in failed)
    return KPI_PENALTY * len(failed) + VIOLATION_WEIGHT * violation + _normalized_primary(metrics, spec)


def validate_spec_metrics(spec: TaskSpec) -> list[str]:
    """Return problems with the spec's metric choices (empty list = fine)."""
    problems = []
    names = METRIC_REGISTRY.get(spec.prediction_format, {})
    for name in {spec.primary_metric, *spec.metrics, *(k.metric for k in spec.kpis)}:
        if name not in names:
            problems.append(f"metric '{name}' is not in the registry for format '{spec.prediction_format}'")
    return problems


def compute_metrics(fmt: PredictionFormat, predictions_path: str, labels_path: str, metric_names: list[str]):
    """TODO(ML): load predictions + hidden labels, compute each requested metric for this format."""
    raise NotImplementedError("TODO(ML): scoring.compute_metrics")
