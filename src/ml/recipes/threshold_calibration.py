"""Recipe: pick the alarm threshold from the task's KPIs instead of guessing 0.5.

binary_score: maximise recall subject to false_alarm_rate <= the KPI (or precision / recall KPIs).
scalar early warning: the alarm rule is fixed (pred <= early_warning_horizon), so instead shift the
prediction by a calibrated offset that makes early_warning_far meet its KPI.
Always calibrate on an INTERNAL split (harness.groups), never on the full training set.
"""

import numpy as np


def kpi(task: dict, metric: str):
    return next((k["threshold"] for k in task["kpis"] if k["metric"] == metric), None)


def calibrate_threshold(scores, labels, max_far=None, min_recall=None) -> float:
    """With max_far: the highest-recall threshold whose FAR <= max_far.
    With only min_recall: the highest threshold (fewest false alarms) whose recall >= min_recall."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    best = float(np.max(scores)) + 1e-9
    best_recall = -1.0
    for t in np.unique(scores):  # ascending
        alarm = scores >= t
        far = (alarm & ~labels).sum() / max((~labels).sum(), 1)
        recall = (alarm & labels).sum() / max(labels.sum(), 1)
        if max_far is not None:
            if far <= max_far and recall > best_recall:
                best, best_recall = float(t), recall
        elif min_recall is not None and recall >= min_recall:
            best = float(t)  # keeps the last (highest) threshold that still reaches min_recall
    return best


# Usage inside train.py (binary_score), after fitting `pipe` on the internal-train part:
#   task = harness.task()
#   t = calibrate_threshold(pipe.predict_proba(X_va)[:, 1], y_va,
#                           max_far=kpi(task, "false_alarm_rate"), min_recall=kpi(task, "recall"))
#   print(f"threshold={t:.3f}")
#   harness.finish(predict_fn, threshold=t)
#
# Scalar early warning (e.g. RUL): find the smallest upward shift s so that
#   mean(pred_va + s <= H for units with y_va > H) <= kpi(task, "early_warning_far"),
# then make the deployed model predict pred + s (wrap it, e.g. sklearn TransformedTargetRegressor or a
# tiny Pipeline step), because the deployed model's output must already include the shift.

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.random(1000) < 0.1
    s = np.clip(y * 0.4 + rng.random(1000) * 0.7, 0, 1)
    t = calibrate_threshold(s, y, max_far=0.05)
    print(f"threshold={t:.3f} recall={(s[y] >= t).mean():.3f} far={(s[~y] >= t).mean():.3f}")
