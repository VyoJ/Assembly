"""Threshold calibration: many KPIs (false-alarm rate, recall) are won or lost here.

TODO(ML): implement.
"""

def calibrate_threshold(scores, labels, max_far: float | None = None, min_recall: float | None = None) -> float:
    """Pick the threshold meeting the constraint(s) on an internal split; prefer max recall under max_far."""
    raise NotImplementedError("TODO(ML): models.calibrate.calibrate_threshold")
