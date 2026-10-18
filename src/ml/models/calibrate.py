"""Threshold calibration: many KPIs (false-alarm rate, recall) are won or lost here.

Convention (same as scoring): a sample is an alarm when score >= threshold.
Calibrate on an internal hold-out (models.splits), never on the data the model was fit on.
"""

import numpy as np


def _rates(scores: np.ndarray, labels: np.ndarray):
    """Recall and false-alarm rate for every candidate threshold (unique scores, high to low)."""
    order = np.argsort(-scores, kind="stable")
    s, lab = scores[order], labels[order]
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    last = np.r_[s[1:] != s[:-1], True]  # one point per distinct score
    pos, neg = max(lab.sum(), 1), max((1 - lab).sum(), 1)
    return s[last], tp[last] / pos, fp[last] / neg


def calibrate_threshold(scores, labels, max_far: float | None = None, min_recall: float | None = None) -> float:
    """Pick the threshold meeting the constraint(s) on an internal split; prefer max recall under max_far.

    max_far only:     highest recall with false-alarm rate <= max_far
    min_recall only:  lowest false-alarm rate with recall >= min_recall
    both:             lowest false-alarm rate among thresholds meeting both; if none, best effort on max_far
    neither:          threshold with the best F1
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        raise ValueError("calibration needs both positive and negative labels in the hold-out")
    thr, recall, far = _rates(scores, labels)
    # Candidate "no alarms at all": threshold just above the max score.
    thr = np.r_[np.nextafter(thr[0], np.inf), thr]
    recall, far = np.r_[0.0, recall], np.r_[0.0, far]

    ok_far = far <= max_far if max_far is not None else np.ones_like(far, bool)
    ok_rec = recall >= min_recall if min_recall is not None else np.ones_like(recall, bool)

    if max_far is not None and min_recall is not None and (ok_far & ok_rec).any():
        cand = np.flatnonzero(ok_far & ok_rec)
        return float(thr[cand[np.argmin(far[cand])]])
    if max_far is not None:
        cand = np.flatnonzero(ok_far)
        return float(thr[cand[np.argmax(recall[cand])]])
    if min_recall is not None:
        cand = np.flatnonzero(ok_rec)
        return float(thr[cand[np.argmin(far[cand])]])

    precision = np.divide(
        recall * labels.sum(), recall * labels.sum() + far * (1 - labels).sum(),
        out=np.zeros_like(recall), where=recall > 0,
    )
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    return float(thr[np.argmax(f1)])
