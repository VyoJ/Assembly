"""Models for signals. TODO."""
from ..registry import register_model


@register_model("timeseries", "classification", "xgboost")
def xgb_classifier(config):
    raise NotImplementedError


@register_model("timeseries", "regression", "xgboost")
def xgb_regressor(config):
    raise NotImplementedError


@register_model("timeseries", "anomaly_detection", "cnn1d")
def cnn1d_anomaly(config):
    raise NotImplementedError
