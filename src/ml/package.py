"""Winning experiment -> deployable package + Predictor.

{ws}/package/
    model/           copied from the winner's model/ (see ml.models.export)
    manifest.json    task, format, threshold, alarm rule, feature columns, val + test metrics, runtime provider
    reference.json   per-feature training medians, used to fill features a live sample doesn't carry
    replay.parquet   held-out test inputs the live demo can replay through the model (tabular tasks)
"""

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.contracts import Workspace
from ml import runner
from ml.models import export

REPLAY_ROWS = 2000


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def package(ws: Workspace) -> str:
    """Copy the winner's model + preprocess/threshold info into {ws}/package, write manifest.json, return path."""
    import pandas as pd

    wsp = Path(ws.path)
    src = runner.winner_dir(ws)
    if not (src / "model" / "model.json").exists():
        raise RuntimeError(f"{ws.task_id}: {src.name}/model/model.json missing; train.py must call harness.save_model()")

    pkg = wsp / "package"
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(src / "model", pkg / "model")
    model_info = _read_json(pkg / "model" / "model.json")
    data_meta = _read_json(wsp / "data" / "meta.json")
    spec = ws.spec

    feature_columns = model_info.get("feature_columns")
    train_inputs = wsp / "data" / "train_inputs.parquet"
    if spec.prediction_format not in ("boxes", "masks") and train_inputs.exists():
        X = pd.read_parquet(train_inputs)
        X = X.drop(columns=["sample_id"], errors="ignore")
        feature_columns = feature_columns or [str(c) for c in X.columns]
        num = X.select_dtypes("number")
        (pkg / "reference.json").write_text(json.dumps({str(c): float(v) for c, v in num.median().items()}, indent=2))
        test_inputs = wsp / "data" / "test_inputs.parquet"
        if test_inputs.exists():
            pd.read_parquet(test_inputs).head(REPLAY_ROWS).to_parquet(pkg / "replay.parquet", index=False)

    val = _read_json(src / "result.json")
    manifest = {
        "task_id": ws.task_id,
        "task_name": spec.name,
        "exp_id": "full_train" if src.name == "full" else src.name,
        "prediction_format": spec.prediction_format,
        "primary_metric": spec.primary_metric,
        "kpis": [k.model_dump() for k in spec.kpis],
        "threshold": _read_json(src / "predict_meta.json").get("threshold", model_info.get("threshold")),
        "early_warning_horizon": data_meta.get("early_warning_horizon"),
        "classes": data_meta.get("classes") or model_info.get("classes"),
        "feature_columns": feature_columns,
        "model_kind": model_info.get("kind"),
        "onnx": model_info.get("onnx"),
        "val_metrics": val.get("metrics", {}),
        "test_metrics": _read_json(wsp / "final_report.json").get("test_metrics", {}),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    pred = load_predictor(str(pkg))  # fail now, not on the live stream
    if pred.n_replay:
        pred.predict(pred.replay(0))
    manifest["runtime_provider"] = pred.provider
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return str(pkg)


class Predictor:
    """.predict(sample) -> dict shaped by the prediction format (same keys as ml.mock.MockPredictor):
    scalar {"value", "alarm"}, binary_score {"score", "alarm"}, class_probs {"probs", "label", "alarm"},
    boxes {"boxes", "alarm"}; always "latency_ms".

    sample: {feature_column: value, ...} (or {"features": {...}}); features it doesn't carry are filled with
    training medians. {"replay": i} uses row i of the held-out replay set. Image tasks: {"image_path": ...}.
    """

    def __init__(self, path: str):
        import pandas as pd

        self.path = Path(path)
        self.manifest = _read_json(self.path / "manifest.json")
        self.fmt = self.manifest["prediction_format"]
        self.model = export.load_saved(str(self.path / "model"))
        self.provider = self.model.provider or ("python:" + str(self.model.kind))
        self.features = self.manifest.get("feature_columns") or []
        self.reference = _read_json(self.path / "reference.json")
        t = self.manifest.get("threshold")
        self.threshold = float(t) if t is not None else 0.5
        h = self.manifest.get("early_warning_horizon")
        self.horizon = float(h) if h is not None else 30.0
        rp = self.path / "replay.parquet"
        self._replay = pd.read_parquet(rp) if rp.exists() else None

    @property
    def n_replay(self) -> int:
        return 0 if self._replay is None else len(self._replay)

    def replay(self, i: int) -> dict:
        row = self._replay.iloc[i % self.n_replay].to_dict()
        return {"features": row, "sample_id": row.get("sample_id")}

    def _frame(self, sample: dict):
        import pandas as pd

        if "replay" in sample and self.n_replay:
            sample = self.replay(int(sample["replay"]))
        feats = sample.get("features", sample)
        row = {}
        for c in self.features:
            v = feats.get(c, self.reference.get(c))
            row[c] = np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else v
        X = pd.DataFrame([row], columns=self.features)
        for c in X.columns:
            try:
                X[c] = pd.to_numeric(X[c])
            except (ValueError, TypeError):
                pass
        return X

    def predict(self, sample: dict) -> dict:
        t0 = time.perf_counter()
        if self.fmt in ("boxes", "masks"):
            out = self._predict_image(sample)
        else:
            X = self._frame(sample)
            if self.fmt == "scalar":
                v = float(np.ravel(self.model.predict(X))[0])
                out = {"value": round(v, 3), "alarm": v <= self.horizon}
            elif self.fmt == "binary_score":
                proba = self.model.predict_proba(X)
                s = float(proba[0, -1]) if proba is not None else float(np.ravel(self.model.predict(X))[0])
                out = {"score": round(s, 4), "alarm": s >= self.threshold}
            else:  # class_probs
                proba = self.model.predict_proba(X)
                p = np.ravel(proba[0] if proba is not None else self.model.predict(X)[0]).astype(float)
                classes = self.manifest.get("classes") or list(range(len(p)))
                label = classes[int(np.argmax(p))]
                out = {"probs": [round(x, 4) for x in p.tolist()], "label": label, "alarm": False}
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def _predict_image(self, sample: dict) -> dict:
        path = sample.get("image_path") or sample.get("image")
        if not path:
            return {"boxes": [], "alarm": False, "error": "sample has no image_path"}
        boxes = []
        if self.model.kind == "ultralytics":
            for r in self.model.predict(path):
                for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    boxes.append([*(round(v, 1) for v in xyxy), round(conf, 3), int(cls)])
        else:
            raise NotImplementedError(f"image predictor for model kind {self.model.kind!r}")
        return {"boxes": boxes, "alarm": any(b[4] >= self.threshold for b in boxes)}


def load_predictor(package_path: str) -> Predictor:
    return Predictor(package_path)
