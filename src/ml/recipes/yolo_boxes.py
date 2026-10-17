"""Recipe: defect localisation (prediction_format "boxes") by fine-tuning Ultralytics YOLO11-nano. Runs on CPU.

Converts harness data to a YOLO dataset under the experiment dir (images are symlinked, labels normalised),
trains for a WALL-CLOCK budget (`time=` hours, from harness.time_left()), calibrates the alarm threshold on an
internal split from each image's best box score, and saves the best checkpoint (deployed as is).
~4 min on a laptop CPU for MVTec metal_nut at 320 px; image-level AUROC ~0.99, box mAP50 ~0.7.

Ultralytics never touches the network here: offline mode, no plots (they download a font), no AMP check.
Ideas to try next: imgsz 416-640 for small defects (slower), yolo11s.pt if listed in data_card.md, more time.
"""

import os

os.environ["YOLO_OFFLINE"] = "1"  # before importing ultralytics: no update checks / downloads / telemetry

from pathlib import Path

import harness
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from ultralytics import YOLO

from ml.models.calibrate import calibrate_threshold

IMGSZ = 320
task = harness.task()
X, y = harness.load_train()  # X: DataFrame with `path`; y: per image, a list of [x1, y1, x2, y2, class]
classes = harness.classes() or ["defect"]
exp = Path(os.environ["ASSEMBLY_EXP_DIR"])


def kpi(metric):
    return next((k["threshold"] for k in task["kpis"] if k["metric"] == metric), None)


def write_yolo_split(name, rows):
    images, labels = exp / "yolo" / name / "images", exp / "yolo" / name / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for sid in rows:
        src = Path(X.loc[sid, "path"])
        link = images / f"{sid}{src.suffix}"
        if not link.exists():
            link.symlink_to(src)
        w, h = Image.open(src).size
        lines = [f"{int(b[4])} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} "
                 f"{(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}" for b in y.loc[sid]]
        (labels / f"{sid}.txt").write_text("\n".join(lines))


has_box = np.array([len(b) > 0 for b in y])
tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, has_box))
write_yolo_split("train", X.index[tr])
write_yolo_split("val", X.index[va])
names = "{" + ", ".join(f"{i}: {c}" for i, c in enumerate(classes)) + "}"
(exp / "yolo" / "data.yaml").write_text(f"path: {exp / 'yolo'}\ntrain: train/images\nval: val/images\nnames: {names}\n")

hours = max(30.0, 0.75 * harness.time_left() - 60) / 3600  # seconds -> hours; leave time to calibrate + export
print(f"train_hours={hours:.3f} n_train={len(tr)} n_internal_val={len(va)}")
model = YOLO(harness.weights("yolo11n.pt"))
model.train(data=str(exp / "yolo" / "data.yaml"), epochs=300, time=hours, imgsz=IMGSZ, batch=16,
            device=harness.device() if harness.device() != "mps" else "cpu", workers=0, plots=False,
            amp=False, verbose=False, project=str(exp / "yolo" / "runs"), name="train", exist_ok=True, patience=50)
model = YOLO(str(model.trainer.best))


def predict(X):
    out = []
    for path in X["path"]:
        r = model.predict(path, imgsz=IMGSZ, conf=0.01, max_det=20, verbose=False)[0]
        rows = zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True)
        out.append([[*xyxy, float(c), int(k)] for xyxy, c, k in rows])
    return out


# Alarm = any box scoring >= threshold: calibrate it from each internal-val image's best box score.
best = np.array([max((b[4] for b in boxes), default=0.0) for boxes in predict(X.iloc[va])])
far_cap = kpi("image_far")
threshold = calibrate_threshold(best, has_box[va], max_far=far_cap if far_cap is not None else 0.1)
threshold = float(min(max(threshold, 0.01), 1.0))
print(f"threshold={threshold:.3f} time_left={harness.time_left():.0f}")

harness.save_model(model, threshold=threshold, imgsz=IMGSZ)
harness.finish(predict, threshold=threshold)
