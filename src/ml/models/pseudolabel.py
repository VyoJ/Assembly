"""SAM 3 -> YOLO dataset conversion, cached by (prompts, score_thr): SAM 3 is slow.

TODO(ML): implement.
"""

def sam3_to_yolo(image_paths: list[str], prompts: list[str], score_thr: float, out_dir: str) -> str:
    """Run SAM 3 text-prompted segmentation, convert masks to YOLO boxes, write data.yaml; return its path. Cache hits are instant."""
    raise NotImplementedError("TODO(ML): models.pseudolabel.sam3_to_yolo")
