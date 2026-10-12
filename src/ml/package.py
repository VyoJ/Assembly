"""Best experiment -> deployable package + Predictor.

TODO(ML): implement. A package dir holds the exported model (ONNX / joblib / SAM 3 config),
preprocess info, and manifest.json. `Dockerfile.runtime` (onnxruntime + FastAPI) builds a
lightweight container from it.
"""

from core.contracts import Workspace


def package(ws: Workspace) -> str:
    """TODO(ML): copy best/model (+ preprocess/threshold info) into {ws}/package, write manifest.json, return path."""
    raise NotImplementedError("TODO(ML): package.package")


def load_predictor(package_path: str):
    """TODO(ML): return an object with .predict(sample: dict) -> dict shaped by the prediction format
    (e.g. {"score", "alarm", "latency_ms"} or {"boxes", "alarm", "latency_ms"})."""
    raise NotImplementedError("TODO(ML): package.load_predictor")
