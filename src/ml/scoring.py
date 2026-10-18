"""Fixed evaluation core: prediction formats, metric registry, KPI checks, scalar score.

The agent may only choose metrics/KPIs that exist here. Score is LOWER = BETTER and
meeting all KPIs always beats missing any.

`compute_metrics` reads the files described in ML_PLAN.md (predictions and hidden labels, matched by sample_id).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

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


# ---------------------------------------------------------------- loading (file layout: ML_PLAN.md)

class PredictionError(ValueError):
    """Predictions missing or not in the task's prediction_format -> experiment status 'invalid_output'."""


DEFAULT_EARLY_WARNING_HORIZON = 30.0
DEFAULT_THRESHOLD = 0.5


def find_file(base: str | Path) -> Path | None:
    """`exp_000/val_predictions` -> the .parquet or .json file that exists, if any."""
    base = Path(base)
    if base.suffix in (".parquet", ".json") and base.exists():
        return base
    for ext in (".parquet", ".json"):
        p = base.with_name(base.name + ext)
        if p.exists():
            return p
    return None


def _read_table(path: Path, columns: list[str] | None = None):
    import pandas as pd

    if path.suffix == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_parquet(path)
    if "sample_id" not in df.columns:
        raise PredictionError(f"{path.name}: missing 'sample_id' column")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise PredictionError(f"{path.name}: duplicate sample_id values")
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise PredictionError(f"{path.name}: missing column(s) {missing}; have {list(df.columns)}")
    return df.set_index("sample_id")


def _read_json_map(path: Path) -> dict[str, list]:
    if path.suffix != ".json":
        raise PredictionError(f"{path.name}: boxes/masks must be a JSON file {{sample_id: [...]}}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise PredictionError(f"{path.name}: expected a JSON object {{sample_id: [...]}}")
    return {str(k): v for k, v in data.items()}


def _aligned(pred, labels, pred_name: str):
    """Reindex predictions onto the label ids; every labelled sample needs exactly one prediction."""
    missing = labels.index.difference(pred.index)
    if len(missing):
        raise PredictionError(f"{pred_name}: no prediction for {len(missing)} sample_id(s), e.g. {list(missing[:3])}")
    return pred.loc[labels.index]


def _finite(a, name: str):
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise PredictionError(f"{name}: contains NaN or inf")
    return a


# ---------------------------------------------------------------- metric implementations (array in, float out)
# Shared with ml.models.selfcheck so the agent's internal checks use the same definitions.

def nasa_score(y, pred) -> float:
    """C-MAPSS scoring function, summed over samples: late predictions (pred > y) cost more."""
    d = np.asarray(pred, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def _rates(y_true, alarm) -> dict[str, float]:
    y_true, alarm = np.asarray(y_true, bool), np.asarray(alarm, bool)
    tp, fp = int(np.sum(alarm & y_true)), int(np.sum(alarm & ~y_true))
    fn, tn = int(np.sum(~alarm & y_true)), int(np.sum(~alarm & ~y_true))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "false_alarm_rate": far}


def auroc(y_true, score) -> float:
    """Rank-based (Mann-Whitney) AUROC with tie handling."""
    import pandas as pd

    y_true, score = np.asarray(y_true, bool), np.asarray(score, float)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, score) -> float:
    """PR-AUC as average precision (step-wise, like sklearn)."""
    y_true, score = np.asarray(y_true, bool), np.asarray(score, float)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    y, s = y_true[order], score[order]
    tp = np.cumsum(y)
    last = np.r_[np.diff(s) != 0, True]  # last index of each distinct score
    tp, k = tp[last], (np.arange(1, len(y) + 1))[last]
    precision, recall = tp / k, tp / y_true.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def scalar_metrics(y, pred, horizon: float = DEFAULT_EARLY_WARNING_HORIZON) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ew = _rates(y <= horizon, pred <= horizon)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else 0.0,
        "nasa_score": nasa_score(y, pred),
        "early_warning_recall": ew["recall"],
        "early_warning_far": ew["false_alarm_rate"],
    }


def binary_metrics(y, score, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    y, score = np.asarray(y, float).astype(bool), np.asarray(score, float)
    return {"auroc": auroc(y, score), "pr_auc": average_precision(y, score), **_rates(y, score >= threshold)}


def class_metrics(y, probs) -> dict[str, float]:
    y, probs = np.asarray(y, int), np.asarray(probs, float)
    pred = probs.argmax(axis=1)
    recalls, f1s = [], []
    for c in range(probs.shape[1]):
        r = _rates(y == c, pred == c)
        if np.any(y == c):
            recalls.append(r["recall"])
        f1s.append(r["f1"])
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1s)),
        "min_class_recall": float(min(recalls)) if recalls else 0.0,
    }


def _box_iou(a, b) -> np.ndarray:
    """IoU matrix between [N,4] and [M,4] xyxy boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _ap_at(gts: dict, preds: dict, iou_thr: float, iou_fn) -> float:
    """COCO-style AP (101-point interpolation) averaged over classes present in the ground truth.
    gts: {sid: [(cls, geom)]}, preds: {sid: [(cls, score, geom)]}."""
    classes = sorted({c for items in gts.values() for c, _ in items})
    aps = []
    for c in classes:
        n_gt = sum(1 for items in gts.values() for k, _ in items if k == c)
        dets = sorted(
            ((s, sid, g) for sid, items in preds.items() for k, s, g in items if k == c), key=lambda d: -d[0]
        )
        used = {sid: np.zeros(sum(1 for k, _ in gts.get(sid, []) if k == c), bool) for sid in gts}
        tp = np.zeros(len(dets))
        for i, (_, sid, g) in enumerate(dets):
            gt_geoms = [gg for k, gg in gts.get(sid, []) if k == c]
            if not gt_geoms:
                continue
            ious = iou_fn([g], gt_geoms)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not used[sid][j]:
                used[sid][j] = True
                tp[i] = 1
        if not dets:
            aps.append(0.0)
            continue
        ctp = np.cumsum(tp)
        recall = ctp / n_gt
        precision = ctp / np.arange(1, len(dets) + 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        pts = np.linspace(0, 1, 101)
        idx = np.searchsorted(recall, pts, side="left")
        aps.append(float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx])))
    return float(np.mean(aps)) if aps else 0.0


def _image_flags(gts: dict, preds: dict, threshold: float) -> dict[str, float]:
    ids = list(gts)
    defective = np.array([len(gts[s]) > 0 for s in ids])
    flagged = np.array([any(s_ >= threshold for _, s_, _ in preds.get(s, [])) for s in ids])
    r = _rates(defective, flagged)
    return {"image_recall": r["recall"], "image_far": r["false_alarm_rate"]}


def box_metrics(gt_map: dict, pred_map: dict, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """gt_map: {sid: [[x1,y1,x2,y2,cls], ...]}, pred_map: {sid: [[x1,y1,x2,y2,score,cls], ...]}."""
    gts = {sid: [(int(b[4]), b[:4]) for b in boxes] for sid, boxes in gt_map.items()}
    preds = {sid: [(int(b[5]), float(b[4]), b[:4]) for b in pred_map.get(sid, [])] for sid in gt_map}
    return {
        "map50": _ap_at(gts, preds, 0.5, _box_iou),
        "map50_95": float(np.mean([_ap_at(gts, preds, t, _box_iou) for t in np.arange(0.5, 0.96, 0.05)])),
        **_image_flags(gts, preds, threshold),
    }


def _load_mask(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 127


def _mask_iou(a: list, b: list) -> np.ndarray:
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            union = np.logical_or(x, y).sum()
            out[i, j] = np.logical_and(x, y).sum() / union if union else 0.0
    return out


def mask_metrics(gt_map: dict, pred_map: dict, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """gt_map: {sid: mask_png | null (good part)}, pred_map: {sid: [{"mask": png, "score": s}, ...]}.
    Single defect class; the whole ground-truth mask counts as one instance."""
    gts, preds, ious = {}, {}, []
    for sid, path in gt_map.items():
        gt = _load_mask(path) if path else None
        gts[sid] = [(0, gt)] if gt is not None and gt.any() else []
        preds[sid] = [(0, float(p["score"]), _load_mask(p["mask"])) for p in pred_map.get(sid, [])]
        if gts[sid]:
            kept = [m for _, s, m in preds[sid] if s >= threshold]
            union_pred = np.logical_or.reduce(kept) if kept else np.zeros_like(gts[sid][0][1])
            ious.append(_mask_iou([union_pred], [gts[sid][0][1]])[0, 0])
    return {
        "mask_map": _ap_at(gts, preds, 0.5, _mask_iou),
        "iou": float(np.mean(ious)) if ious else 0.0,
        **_image_flags(gts, preds, threshold),
    }


# ---------------------------------------------------------------- entry point used by ml.runner

def compute_metrics(
    fmt: PredictionFormat,
    predictions_path: str,
    labels_path: str,
    metric_names: list[str],
    params: dict | None = None,
) -> dict[str, float]:
    """Load predictions + hidden labels (matched by sample_id), compute the requested metrics.

    predictions_path / labels_path: file, or base path without extension (.parquet / .json is found).
    params: {"threshold", "early_warning_horizon", "p50_latency_ms", "p95_latency_ms"}; the runner fills
    it from predict_meta.json, data/meta.json and timings.json.
    Raises PredictionError if the predictions don't match the format (-> status 'invalid_output').
    """
    params = params or {}
    pred_file, label_file = find_file(predictions_path), find_file(labels_path)
    if pred_file is None:
        raise PredictionError(f"no predictions file at {predictions_path}(.parquet|.json); did train.py call harness.finish()?")
    if label_file is None:
        raise FileNotFoundError(f"hidden labels not found at {labels_path}")
    threshold = float(params.get("threshold") if params.get("threshold") is not None else DEFAULT_THRESHOLD)

    if fmt == "scalar":
        labels = _read_table(label_file, ["y"])
        pred = _aligned(_read_table(pred_file, ["pred"]), labels, pred_file.name)
        horizon = float(params.get("early_warning_horizon") or DEFAULT_EARLY_WARNING_HORIZON)
        out = scalar_metrics(labels["y"], _finite(pred["pred"], "pred"), horizon)
    elif fmt == "binary_score":
        labels = _read_table(label_file, ["y"])
        pred = _aligned(_read_table(pred_file, ["score"]), labels, pred_file.name)
        out = binary_metrics(labels["y"], _finite(pred["score"], "score"), threshold)
    elif fmt == "class_probs":
        labels = _read_table(label_file, ["y"])
        pred = _read_table(pred_file)
        cols = sorted((c for c in pred.columns if c.startswith("p_")), key=lambda c: int(c[2:]))
        if not cols:
            raise PredictionError(f"{pred_file.name}: expected probability columns p_0 .. p_<k-1>")
        n_classes = int(labels["y"].max()) + 1
        if len(cols) < n_classes:
            raise PredictionError(f"{pred_file.name}: {len(cols)} probability columns but labels have {n_classes} classes")
        pred = _aligned(pred, labels, pred_file.name)
        out = class_metrics(labels["y"], _finite(pred[cols], "class probabilities"))
    elif fmt in ("boxes", "masks"):
        gt_map, pred_map = _read_json_map(label_file), _read_json_map(pred_file)
        missing = set(gt_map) - set(pred_map)
        if missing:
            raise PredictionError(f"{pred_file.name}: no entry for {len(missing)} sample_id(s), e.g. {sorted(missing)[:3]}"
                                  " (use [] for images with no detections)")
        if fmt == "boxes":
            for sid in gt_map:
                for b in pred_map[sid]:
                    if not (isinstance(b, (list, tuple)) and len(b) == 6):
                        raise PredictionError(f"{pred_file.name}[{sid}]: each box must be [x1, y1, x2, y2, score, class]")
            out = box_metrics(gt_map, pred_map, threshold)
        else:
            for sid in gt_map:
                for p in pred_map[sid]:
                    if not (isinstance(p, dict) and "mask" in p and "score" in p):
                        raise PredictionError(f"{pred_file.name}[{sid}]: each entry must be {{\"mask\": png_path, \"score\": s}}")
            out = mask_metrics(gt_map, pred_map, threshold)
    else:
        raise ValueError(f"unknown prediction format {fmt!r}")

    for k in ("p50_latency_ms", "p95_latency_ms"):
        if params.get(k) is not None:
            out[k] = float(params[k])
    wanted = set(metric_names)
    return {k: float(v) for k, v in out.items() if k in wanted}
