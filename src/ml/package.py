"""Winning experiment -> deployable package + Predictor.

{ws}/package/
    model/                    copied from the winner's model/ (see ml.models.export)
    manifest.json             task, format, input kind, threshold, alarm rule, feature columns, metrics, provider
    reference.json            per-feature training medians, to fill features a live sample doesn't carry (tables)
    replay_inputs.parquet     held-out test inputs (tables: features; files: absolute paths)
    replay_predictions.*      the winner's scored test predictions, matched to replay_inputs by sample_id
"""

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.contracts import Workspace
from ml import runner, scoring
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
    input_kind = data_meta.get("input_kind", {}).get("test", "table")
    spec = ws.spec

    feature_columns = model_info.get("feature_columns")
    if input_kind == "table" and (wsp / "data" / "train_inputs.parquet").exists():
        X = pd.read_parquet(wsp / "data" / "train_inputs.parquet").drop(columns=["sample_id"], errors="ignore")
        feature_columns = feature_columns or [str(c) for c in X.columns]
        med = X.select_dtypes("number").median()
        (pkg / "reference.json").write_text(json.dumps({str(c): float(v) for c, v in med.items()}, indent=2))

    test_inputs = wsp / "data" / "test_inputs.parquet"
    if test_inputs.exists():
        replay = pd.read_parquet(test_inputs).head(REPLAY_ROWS)
        if input_kind == "files" and "path" in replay:
            replay["path"] = [str((wsp / "data" / p).resolve()) for p in replay["path"]]
        replay.to_parquet(pkg / "replay_inputs.parquet", index=False)
    preds = scoring.find_file(src / "test_predictions")
    if preds is not None:
        shutil.copy(preds, pkg / f"replay_predictions{preds.suffix}")
    _write_replay_labels(ws, pkg)

    timings = _read_json(src / "timings.json")
    manifest = {
        "task_id": ws.task_id,
        "task_name": spec.name,
        "exp_id": "full_train" if src.name == "full" else src.name,
        "prediction_format": spec.prediction_format,
        "input_kind": input_kind,
        "primary_metric": spec.primary_metric,
        "kpis": [k.model_dump() for k in spec.kpis],
        "threshold": _read_json(src / "predict_meta.json").get("threshold", model_info.get("threshold")),
        "early_warning_horizon": data_meta.get("early_warning_horizon"),
        "classes": data_meta.get("classes") or model_info.get("classes"),
        "normal_class": _normal_class(data_meta.get("classes") or model_info.get("classes"), data_meta),
        "feature_columns": feature_columns,
        "model_kind": model_info.get("kind"),
        "onnx": model_info.get("onnx"),
        "p50_latency_ms": timings.get("p50_latency_ms"),
        "val_metrics": _read_json(src / "result.json").get("metrics", {}),
        "test_metrics": _read_json(wsp / "final_report.json").get("test_metrics", {}),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    pred = load_predictor(str(pkg))  # fail now, not on the live stream
    if pred.n_replay:
        pred.predict({"replay": 0})
    manifest["runtime_provider"] = pred.provider
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return str(pkg)


def _write_replay_labels(ws: Workspace, pkg: Path) -> None:
    """True test labels of the replay rows, so the live dashboard can show hit / miss / false alarm.
    Written after the final test (the agent never reads packages), only for one-value-per-sample labels."""
    import pandas as pd

    from ml import workspace

    hidden = workspace.labels_dir(ws.run_id, ws.task_id)
    replay = pkg / "replay_inputs.parquet"
    if not replay.exists():
        return
    sids = set(pd.read_parquet(replay, columns=["sample_id"])["sample_id"])
    if (hidden / "test_labels.parquet").exists():
        y = pd.read_parquet(hidden / "test_labels.parquet").set_index("sample_id")["y"].to_dict()
    elif (hidden / "test_labels.json").exists():  # boxes: {sample_id: [[x1, y1, x2, y2, class], ...]}
        y = json.loads((hidden / "test_labels.json").read_text())
    else:
        return
    (pkg / "replay_labels.json").write_text(json.dumps({str(k): v for k, v in y.items() if k in sids}, default=float))


NORMAL_NAMES = {"normal", "good", "ok", "healthy", "no_fault", "nominal"}


def _normal_class(classes, data_meta: dict):
    """The class that means 'no problem', for the class_probs alarm: meta.json["normal_class"] if prepare.py
    set it, else a class with a conventional name. None -> class_probs tasks never alarm."""
    if data_meta.get("normal_class") is not None:
        return data_meta["normal_class"]
    return next((c for c in classes or [] if str(c).lower() in NORMAL_NAMES), None)


class Predictor:
    """.predict(sample) -> dict shaped by the prediction format (same keys as ml.mock.MockPredictor):
    scalar {"value", "alarm"}, binary_score {"score", "alarm"}, class_probs {"probs", "label", "alarm"},
    boxes {"boxes", "alarm"}; always "latency_ms".

    sample:
      {"replay": i}                     held-out test sample i: its inputs and the winner's scored prediction
      {feature: value, ...}             tables: runs the model; missing features use training medians
      {"image_path": p} / {"path": p}   files: runs the model on the file
      {"array": [...]}                  array inputs: runs the model on one sample
    """

    def __init__(self, path: str):
        import pandas as pd

        self.path = Path(path)
        self.manifest = _read_json(self.path / "manifest.json")
        self.fmt = self.manifest["prediction_format"]
        self.input_kind = self.manifest.get("input_kind", "table")
        self.model = export.load_saved(str(self.path / "model"))
        self.provider = self.model.provider or ("python:" + str(self.model.kind))
        self.features = self.manifest.get("feature_columns") or []
        self.reference = _read_json(self.path / "reference.json")
        t = self.manifest.get("threshold")
        self.threshold = float(t) if t is not None else 0.5
        h = self.manifest.get("early_warning_horizon")
        self.horizon = float(h) if h is not None else 30.0
        self.classes = self.manifest.get("classes")
        self.normal_class = self.manifest.get("normal_class")
        rp = self.path / "replay_inputs.parquet"
        self._replay = pd.read_parquet(rp) if rp.exists() else None
        self._stored = self._load_stored()
        self._truth = _read_json(self.path / "replay_labels.json")

    def replay_paths(self) -> dict:
        """sample_id -> absolute file path of the replay inputs (file tasks only)."""
        if self._replay is None or "path" not in self._replay:
            return {}
        return dict(zip(self._replay["sample_id"], self._replay["path"], strict=True))

    def truth(self, sample_id: str):
        """True label of a replay sample (class name for class_probs), or None if unknown."""
        y = self._truth.get(sample_id)
        if y is not None and self.fmt == "class_probs" and self.classes:
            return self.classes[int(y)]
        return y

    def _load_stored(self) -> dict:
        p = scoring.find_file(self.path / "replay_predictions")
        if p is None:
            return {}
        if p.suffix == ".json":
            return json.loads(p.read_text())
        import pandas as pd

        df = pd.read_parquet(p).set_index("sample_id")
        if self.fmt == "class_probs":
            cols = sorted((c for c in df.columns if c.startswith("p_")), key=lambda c: int(c[2:]))
            return {sid: row.tolist() for sid, row in zip(df.index, df[cols].to_numpy(), strict=True)}
        col = "pred" if self.fmt == "scalar" else "score"
        return df[col].to_dict()

    @property
    def n_replay(self) -> int:
        return 0 if self._replay is None else len(self._replay)

    def replay(self, i: int) -> dict:
        """Inputs of held-out test sample i (for showing on screen)."""
        row = self._replay.iloc[i % self.n_replay].to_dict()
        return {"sample_id": row.pop("sample_id", None), **row}

    # ------------------------------------------------------------ output shaping

    def _shape(self, raw) -> dict:
        if self.fmt == "scalar":
            v = float(raw)
            return {"value": round(v, 3), "alarm": v <= self.horizon}
        if self.fmt == "binary_score":
            s = float(raw)
            return {"score": round(s, 4), "alarm": s >= self.threshold}
        if self.fmt == "class_probs":
            p = np.ravel(np.asarray(raw, float))
            names = self.classes or list(range(len(p)))
            label = names[int(np.argmax(p))]
            alarm = self.normal_class is not None and label != self.normal_class
            return {"probs": [round(x, 4) for x in p.tolist()], "label": label, "alarm": alarm}
        boxes = [list(b) for b in raw]
        return {"boxes": boxes, "alarm": any(b[4] >= self.threshold for b in boxes)}

    # ------------------------------------------------------------ inference

    def predict(self, sample: dict) -> dict:
        t0 = time.perf_counter()
        if "replay" in sample and self.n_replay:
            s = self.replay(int(sample["replay"]))
            sid = s["sample_id"]
            if sid in self._stored:
                out = {**self._shape(self._stored[sid]), "sample_id": sid, "source": "replay"}
                out["latency_ms"] = self.manifest.get("p50_latency_ms") or 0.0
                return out
            sample = {"features": s, "image_path": s.get("path")}
        if self.input_kind == "files" or self.fmt in ("boxes", "masks"):
            out = self._predict_file(sample)
        elif self.input_kind == "array":
            out = self._shape(self._raw_output(np.asarray(sample["array"], dtype=np.float32)[None]))
        else:
            out = self._shape(self._raw_output(self._frame(sample)))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def _raw_output(self, X):
        """One sample's model output in the task's format (before alarm / label shaping)."""
        if self.fmt in ("binary_score", "class_probs"):
            proba = self.model.predict_proba(X)
            if proba is not None:
                return proba[0, -1] if self.fmt == "binary_score" else proba[0]
        out = np.asarray(self.model.predict(X))
        return np.ravel(out)[0] if self.fmt in ("scalar", "binary_score") else out[0]

    def _frame(self, sample: dict):
        import pandas as pd

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

    def _predict_file(self, sample: dict) -> dict:
        path = sample.get("image_path") or sample.get("path")
        if not path:
            return {**self._shape([] if self.fmt == "boxes" else 0.0), "error": "sample has no image_path"}
        if self.model.kind == "ultralytics":
            # keep every box the calibrated threshold could alarm on (YOLO's default cut-off is 0.25)
            r = self.model.obj.predict(path, conf=min(0.25, self.threshold), verbose=False)[0]
            if self.fmt == "boxes":
                b = r.boxes
                rows = zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist(), strict=True)
                return self._shape([[*(round(v, 1) for v in xyxy), round(c, 3), int(k)] for xyxy, c, k in rows])
            probs = r.probs.data.cpu().numpy()  # YOLO classification
            return self._shape(probs[-1] if self.fmt == "binary_score" else probs)
        from ml.catalog import load_image, load_signal

        suffix = Path(path).suffix.lower()
        arr = load_image(path) if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff") else load_signal(path)
        return self._shape(self._raw_output(np.asarray(arr, dtype=np.float32)[None]))


def load_predictor(package_path: str) -> Predictor:
    return Predictor(package_path)
