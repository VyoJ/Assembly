"""Vision glue.

TODO(ML): implement.
"""

def masks_to_boxes(mask) -> list[list[float]]:
    """Instance/defect mask -> [x1, y1, x2, y2] boxes."""
    raise NotImplementedError("TODO(ML): models.vision_utils.masks_to_boxes")

def tile_images(images, tile: int, overlap: int = 0):
    """Split images into tiles for small-defect detection."""
    raise NotImplementedError("TODO(ML): models.vision_utils.tile_images")

def untile_preds(preds, tiles_meta):
    """Map tile predictions back to full-image coordinates (+ NMS)."""
    raise NotImplementedError("TODO(ML): models.vision_utils.untile_preds")

def image_level_score(detections) -> float:
    """Max detection score per image (for image_recall / image_far)."""
    raise NotImplementedError("TODO(ML): models.vision_utils.image_level_score")
