"""Registry metrics on the internal split, so the agent can sanity-check before harness.finish().

TODO(ML): implement.
"""

def selfcheck(prediction_format: str, y_true, y_pred, metric_names: list[str]) -> dict[str, float]:
    """Same metric definitions as ml.scoring (share the implementation)."""
    raise NotImplementedError("TODO(ML): models.selfcheck.selfcheck")
