"""Thin wrapper for SAM 3 (Meta): new model, the LLM may not know its API well. Local weights only.

TODO(ML): implement.
"""

def load_sam3(device: str | None = None):
    """Load SAM 3 from harness.weights('sam3')."""
    raise NotImplementedError("TODO(ML): models.sam3.load_sam3")

def sam3_detect(images, prompts: list[str], score_thr: float = 0.5) -> list[dict]:
    """Per image: {'boxes', 'masks', 'scores', 'labels'} for every instance matching any prompt."""
    raise NotImplementedError("TODO(ML): models.sam3.sam3_detect")

def sam3_predict_fn(prompts: list[str], score_thr: float = 0.5):
    """A predict_fn usable with harness.finish for zero-shot detection."""
    raise NotImplementedError("TODO(ML): models.sam3.sam3_predict_fn")
