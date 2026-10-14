"""Hand-written prepare.py for specs/rul_example.json (reference for the agent and for testing).

    uv run python scripts/run_research.py --spec specs/rul_example.json --prepare specs/rul_example_prepare.py
"""

import harness
import numpy as np

RUL_CAP = 125
SEED = 0

spec = harness.task()
df = harness.read_streams(spec["input_streams"])
failures = harness.read_labels("maintenance_log")

df = df.merge(failures, on="entity")
df["rul"] = (df["event_time"] - df["time"]).clip(upper=RUL_CAP)

# Split by unit (70/15/15) so no machine appears in two splits.
units = np.random.default_rng(SEED).permutation(df["entity"].unique())
n_train, n_val = int(0.7 * len(units)), int(0.15 * len(units))
split_units = {
    "train": units[:n_train],
    "val": units[n_train:n_train + n_val],
    "test": units[n_train + n_val:],
}

features = ["time", *spec["input_streams"]]
parts, groups = {}, {}
for split, us in split_units.items():
    part = df[df["entity"].isin(us)].sort_values(["entity", "time"])
    parts[split] = (part[features], part["rul"])
    groups[split] = part["entity"].tolist()

harness.write_dataset(
    parts["train"], parts["val"], parts["test"],
    groups=groups,
    meta={"split": "group", "group_col": "entity", "time_col": "time", "target": f"RUL clipped at {RUL_CAP}"},
)
