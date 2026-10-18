# Recipe (prepare.py pattern), copied from specs/metal_nut_boxes_example_prepare.py. Borrow the pattern; adapt ids, target and split to YOUR task.
"""Hand-written prepare.py for specs/metal_nut_boxes_example.json (reference for the agent and the tests).

Boxes come from the QC inspection's pixel masks: one box per connected defect region, class 0 ("defect").
Same stratified random split as metal_nut_example (one image per physical part); `source_split` is not used
because the source's training folder only holds good parts.
"""

import harness
import numpy as np
from PIL import Image
from scipy import ndimage

SEED = 0


def mask_to_boxes(mask_path) -> list[list[int]]:
    if not isinstance(mask_path, str):
        return []  # good part
    labelled, _ = ndimage.label(np.asarray(Image.open(mask_path).convert("L")) > 127)
    return [[int(x.start), int(y.start), int(x.stop), int(y.stop), 0] for y, x in ndimage.find_objects(labelled)]


spec = harness.task()
images = harness.read_stream(spec["input_streams"][0])
qc = harness.read_labels(spec["label_sources"][0])  # record_id, label (defect type), defective, mask_path
df = images[["record_id", "path"]].merge(qc[["record_id", "label", "mask_path"]], on="record_id")
df["boxes"] = [mask_to_boxes(m) for m in df["mask_path"]]

rng = np.random.default_rng(SEED)
split_of = {}
for _, rows in df.groupby("label"):  # every defect type in every split
    ids = rng.permutation(rows["record_id"].to_numpy())
    n_train, n_val = round(0.6 * len(ids)), round(0.2 * len(ids))
    split_of.update({r: "train" for r in ids[:n_train]})
    split_of.update({r: "val" for r in ids[n_train:n_train + n_val]})
    split_of.update({r: "test" for r in ids[n_train + n_val:]})
df["split"] = df["record_id"].map(split_of)

parts = {}
for split in ("train", "val", "test"):
    part = df[df["split"] == split].sample(frac=1.0, random_state=SEED)
    parts[split] = (part["path"].tolist(), part["boxes"].tolist())
    print(f"{split}: {len(part)} images, {sum(map(bool, part['boxes']))} with defects, "
          f"{sum(map(len, part['boxes']))} boxes")

harness.write_dataset(parts["train"], parts["val"], parts["test"], meta={"split": "random", "classes": ["defect"]})
