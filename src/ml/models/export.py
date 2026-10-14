"""Every kept experiment must be deployable. Used by harness.save_model (writes) and ml.package (reads).

Rule (program.md): the saved model takes the RAW feature columns of harness.load_*_inputs() (sample_id is
the index, not a feature) and returns predictions in the task's format. Put preprocessing inside the model
(sklearn Pipeline, torch module) so deployment is just model(X).

out_dir/
    model.onnx | model.joblib | model.pt (+ model.onnx when the ONNX export works)
    model.json   {"kind", "file", "onnx"?, "feature_columns"?, ...json-serialisable save_model kwargs}
"""

import json
import shutil
from pathlib import Path

import numpy as np

# Preferred ONNX Runtime providers, AMD first: MIGraphX on Instinct/Radeon, then ROCm, then CPU.
ORT_PROVIDERS = ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def _jsonable(kwargs: dict) -> tuple[dict, dict]:
    ok, rest = {}, {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        try:
            json.dumps(v)
            ok[k] = v
        except TypeError:
            rest[k] = v
    return ok, rest


def _is_ultralytics(model) -> bool:
    return type(model).__module__.startswith("ultralytics")


def _is_torch(model) -> bool:
    try:
        import torch

        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False


def _example_array(example_input):
    if example_input is None:
        return None
    if hasattr(example_input, "to_numpy"):  # pandas
        return example_input.to_numpy(dtype=np.float32)
    return np.asarray(example_input, dtype=np.float32)


def _tree_or_sklearn_to_onnx(model, example: np.ndarray, path: Path) -> bool:
    """Best effort: skl2onnx / onnxmltools are optional. joblib stays the source of truth."""
    n = example.shape[1]
    try:
        mod = type(model).__module__
        if mod.startswith(("xgboost", "lightgbm")):
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType

            conv = onnxmltools.convert_xgboost if mod.startswith("xgboost") else onnxmltools.convert_lightgbm
            onx = conv(model, initial_types=[("input", FloatTensorType([None, n]))])
        else:
            from skl2onnx import to_onnx

            onx = to_onnx(model, example[:1])
        path.write_bytes(onx.SerializeToString())
        return True
    except Exception:  # noqa: BLE001 - converters are optional and partial
        return False


def save_model(model, out_dir: str, example_input=None, **kwargs) -> str:
    """ultralytics YOLO -> its own ONNX export (+ .pt); torch nn.Module -> ONNX (TorchScript fallback);
    sklearn / xgboost / lightgbm / anything picklable -> joblib (+ ONNX if a converter is installed).
    kwargs (e.g. feature_columns, threshold, classes) go into model.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info, extra = _jsonable(kwargs)
    example = _example_array(example_input)
    if example_input is not None and hasattr(example_input, "columns"):
        info.setdefault("feature_columns", [str(c) for c in example_input.columns])

    if _is_ultralytics(model):
        info["kind"] = "ultralytics"
        pt = getattr(model, "ckpt_path", None) or getattr(model, "pt_path", None)
        if pt and Path(pt).exists():
            shutil.copy(pt, out / "model.pt")
        else:
            model.save(str(out / "model.pt"))
        info["file"] = "model.pt"
        try:
            onnx_path = model.export(format="onnx", imgsz=kwargs.get("imgsz", 640), dynamic=True)
            shutil.copy(onnx_path, out / "model.onnx")
            info["onnx"] = "model.onnx"
        except Exception as e:  # noqa: BLE001
            info["onnx_error"] = repr(e)[:300]
    elif _is_torch(model):
        import torch

        if example is None:
            raise ValueError("save_model(torch module) needs example_input=<a batch of inputs> to export ONNX")
        model = model.eval().cpu()
        x = torch.from_numpy(np.array(example[: min(len(example), 8)], copy=True))
        try:
            torch.onnx.export(
                model, (x,), str(out / "model.onnx"), input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False,
            )
            info.update(kind="onnx", file="model.onnx", onnx="model.onnx")
        except Exception as e:  # noqa: BLE001
            torch.jit.trace(model, x).save(str(out / "model.ts"))
            info.update(kind="torchscript", file="model.ts", onnx_error=repr(e)[:300])
    else:
        import joblib

        joblib.dump(model, out / "model.joblib")
        info.update(kind="joblib", file="model.joblib")
        if example is not None and example.ndim == 2 and _tree_or_sklearn_to_onnx(model, example, out / "model.onnx"):
            info["onnx"] = "model.onnx"

    if extra:
        import joblib

        joblib.dump(extra, out / "extra.joblib")
        info["extra_file"] = "extra.joblib"
    (out / "model.json").write_text(json.dumps(info, indent=2))
    return str(out)


class SavedModel:
    """Uniform wrapper: .predict(X) -> raw model output as numpy; .predict_proba(X) when the model has it."""

    def __init__(self, path: Path, info: dict, obj, session=None):
        self.path, self.info, self.obj, self.session = path, info, obj, session
        self.kind = info.get("kind")
        self.provider = session.get_providers()[0] if session is not None else None

    def predict(self, X):
        if self.kind == "ultralytics":
            return self.obj.predict(X, verbose=False)
        if self.session is not None:
            name = self.session.get_inputs()[0].name
            return np.asarray(self.session.run(None, {name: _example_array(X)})[0])
        if self.kind == "torchscript":
            import torch

            with torch.no_grad():
                return self.obj(torch.from_numpy(_example_array(X))).numpy()
        return np.asarray(self.obj.predict(X))

    def predict_proba(self, X):
        if self.kind == "joblib" and hasattr(self.obj, "predict_proba"):
            return np.asarray(self.obj.predict_proba(X))
        return None

    @property
    def extra(self) -> dict:
        if "extra_file" not in self.info:
            return {}
        import joblib

        return joblib.load(self.path / self.info["extra_file"])


def ort_session(onnx_path: Path):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = [p for p in ORT_PROVIDERS if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def load_saved(path: str) -> SavedModel:
    """Inverse of save_model; used by ml.package.load_predictor.
    joblib models are served natively (exact same predictions as in training); torch models via ONNX Runtime
    with the AMD providers first."""
    p = Path(path)
    info = json.loads((p / "model.json").read_text())
    kind = info.get("kind")
    if kind == "ultralytics":
        from ultralytics import YOLO

        return SavedModel(p, info, YOLO(str(p / info["file"])))
    if kind == "onnx":
        return SavedModel(p, info, None, ort_session(p / info["file"]))
    if kind == "torchscript":
        import torch

        return SavedModel(p, info, torch.jit.load(str(p / info["file"])))
    import joblib

    return SavedModel(p, info, joblib.load(p / info["file"]))
