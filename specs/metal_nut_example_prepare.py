"""Hand-written prepare.py for specs/metal_nut_example.json (reference for the agent and the tests).

One image per part: a random split stratified by defect type is safe (validators allow split="random" only for
standalone image files). Don't split on the source dataset's own train/test folders: in MVTec the training folder holds only good parts,
so that split would leak the label (the catalog doesn't expose it).
"""

import harness
import numpy as np

SEED = 0

spec = harness.task()
images = harness.read_stream(spec["input_streams"][0])  # record_id, path, ...
qc = harness.read_labels(spec["label_sources"][0])  # record_id, label (defect type), defective
df = images[["record_id", "path"]].merge(qc[["record_id", "label", "defective"]], on="record_id")

rng = np.random.default_rng(SEED)
split_of = {}
for _, rows in df.groupby("label"):  # stratify: every defect type in every split
    ids = rng.permutation(rows["record_id"].to_numpy())
    n_train, n_val = int(round(0.6 * len(ids))), int(round(0.2 * len(ids)))
    split_of.update({r: "train" for r in ids[:n_train]})
    split_of.update({r: "val" for r in ids[n_train:n_train + n_val]})
    split_of.update({r: "test" for r in ids[n_train + n_val:]})
df["split"] = df["record_id"].map(split_of)

parts = {}
for split in ("train", "val", "test"):
    part = df[df["split"] == split].sample(frac=1.0, random_state=SEED)  # no class ordering in the files
    parts[split] = (part["path"].tolist(), part["defective"].astype(int).tolist())
    print(f"{split}: {len(part)} images, {int(part['defective'].sum())} defective")

harness.write_dataset(parts["train"], parts["val"], parts["test"], meta={"split": "random"})
