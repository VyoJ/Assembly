"""Registry metrics on the internal split, so the agent can sanity-check before harness.finish().

Same definitions as ml.scoring (it calls the same functions).
"""

from ml import scoring


def selfcheck(prediction_format: str, y_true, y_pred, metric_names: list[str], **params) -> dict[str, float]:
    """y_pred: scalar -> values, binary_score -> scores, class_probs -> [n, k] probabilities,
    boxes -> {sid: [[x1,y1,x2,y2,score,cls]]} with y_true {sid: [[x1,y1,x2,y2,cls]]}.
    params: threshold=..., early_warning_horizon=..."""
    threshold = params.get("threshold", scoring.DEFAULT_THRESHOLD)
    if prediction_format == "scalar":
        out = scoring.scalar_metrics(
            y_true, y_pred, params.get("early_warning_horizon", scoring.DEFAULT_EARLY_WARNING_HORIZON)
        )
    elif prediction_format == "binary_score":
        out = scoring.binary_metrics(y_true, y_pred, threshold)
    elif prediction_format == "class_probs":
        out = scoring.class_metrics(y_true, y_pred)
    elif prediction_format == "boxes":
        out = scoring.box_metrics(y_true, y_pred, threshold)
    elif prediction_format == "masks":
        out = scoring.mask_metrics(y_true, y_pred, threshold)
    else:
        raise ValueError(f"unknown prediction format {prediction_format!r}")
    return {k: v for k, v in out.items() if k in set(metric_names)}
