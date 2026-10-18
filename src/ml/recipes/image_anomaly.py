"""Recipe: image inspection on CPU with a pretrained ResNet-18 (camera / visual defect tasks).

binary_score (defective or not): PatchKNNAnomaly, PatchCore-style. It learns what NORMAL parts look like (memory
bank of patch features) and scores an image by its most unusual patch. It needs no defect examples: the alarm
threshold comes from normal images the bank never saw (threshold_for_far). ~10 s to fit, ~20-40 ms per image.
class_probs (which defect type): ImageEmbeddingClassifier, pooled embeddings + logistic regression; needs
labelled examples of every class. It also works for binary_score when there are many labelled defects.

Both models take X exactly as harness gives it (DataFrame with `path`) AND decoded image batches (what the
deployed predictor passes), and keep their preprocessing inside, so harness.save_model(model) is all it takes.

Ideas to try next: image_size=320 for small defects, memory_size 20000, an ensemble of both models' ranks,
a class_probs head on top of PatchKNN's score. If the images come from video clips, any internal split must
use harness.groups("train") (neighbouring frames are near-duplicates).
"""

import harness

from ml.models.vision import ImageEmbeddingClassifier, PatchKNNAnomaly

task = harness.task()
fmt = task["prediction_format"]
X, y = harness.load_train()
weights = harness.weights("resnet18.pth")  # cached ImageNet weights; experiments have no network


def kpi(metric):
    return next((k["threshold"] for k in task["kpis"] if k["metric"] == metric), None)


if fmt == "binary_score":
    # Memory bank from 70% of the normal training images (y == 0); the other 30% calibrate the scores.
    model = PatchKNNAnomaly(weights=weights, memory_size=10000, holdout=0.3).fit(X, y)
    # Alarm when a part looks less normal than all but `far` of unseen normal parts. Half the false-alarm KPI
    # keeps a margin: a few held-out images make a noisy quantile. (Calibrating on labelled defects from an
    # internal split was less stable here: few defects, and a model fit on fewer normals scores higher.)
    far_cap = kpi("false_alarm_rate")
    threshold = model.threshold_for_far((far_cap if far_cap is not None else 0.1) / 2)
    print(f"bank={len(model.memory_bank_)} threshold={threshold:.3f} time_left={harness.time_left():.0f}")
    harness.save_model(model, threshold=threshold)
    harness.finish(lambda X: model.predict_proba(X)[:, 1], threshold=threshold)
else:
    model = ImageEmbeddingClassifier(weights=weights).fit(X, y)
    harness.save_model(model)
    harness.finish(model.predict_proba)
