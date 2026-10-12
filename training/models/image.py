"""Modelos de visión sobre backbone preentrenado. TODO."""
from ..registry import register_model


@register_model("image", "classification", "mobilenet_v3")
def mobilenet_classifier(config):
    raise NotImplementedError


@register_model("image", "anomaly_detection", "mobilenet_v3")
def mobilenet_anomaly(config):
    raise NotImplementedError
