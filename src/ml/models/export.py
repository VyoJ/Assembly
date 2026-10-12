"""Every kept experiment must be deployable. Used by harness.save_model.

TODO(ML): implement.
"""

def save_model(model, out_dir: str, example_input=None, **kwargs) -> str:
    """ultralytics YOLO -> model.export(format='onnx'); torch nn.Module -> ONNX (TorchScript fallback); sklearn/xgboost/lightgbm -> joblib; SAM 3 -> config.json with prompts/threshold."""
    raise NotImplementedError("TODO(ML): models.export.save_model")

def load_saved(path: str):
    """Inverse of save_model; used by ml.package.load_predictor."""
    raise NotImplementedError("TODO(ML): models.export.load_saved")
