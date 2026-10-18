# Recipe (prepare.py pattern), copied from specs/rul_example_prepare.py. Borrow the pattern; adapt ids, target and split to YOUR task.
"""Hand-written prepare.py for specs/rul_example.json (reference for the agent and for testing).

    uv run python scripts/run_research.py --spec specs/rul_example.json --prepare specs/rul_example_prepare.py
"""

import harness
import numpy as np

RUL_CAP = 125
SEED = 0
WINDOW = 30  # cycles of history per row

spec = harness.task()
df = harness.read_streams(spec["input_streams"])
failures = harness.read_labels("maintenance_log")

df = df.merge(failures, on="entity")
df["rul"] = (df["event_time"] - df["time"]).clip(upper=RUL_CAP)

# History features, built here because the deployed model sees one row at a time. Causal: each row only uses
# earlier cycles of the same unit (rolling over the past WINDOW cycles, including the current one).
df = df.sort_values(["entity", "time"]).reset_index(drop=True)
sensors = list(spec["input_streams"])
g = df.groupby("entity")[sensors]
roll = g.rolling(WINDOW, min_periods=1)
mean = roll.mean().reset_index(level=0, drop=True).add_suffix(f"_mean{WINDOW}")
std = roll.std().reset_index(level=0, drop=True).fillna(0.0).add_suffix(f"_std{WINDOW}")
past = g.shift(10).fillna(g.transform("first"))  # first 10 cycles: compare with the unit's first cycle
delta = (df[sensors] - past).add_suffix("_delta10")
df = df.join(mean).join(std).join(delta)
history = [*mean.columns, *std.columns, *delta.columns]

# Split by unit (70/15/15) so no machine appears in two splits.
units = np.random.default_rng(SEED).permutation(df["entity"].unique())
n_train, n_val = int(0.7 * len(units)), int(0.15 * len(units))
split_units = {
    "train": units[:n_train],
    "val": units[n_train:n_train + n_val],
    "test": units[n_train + n_val:],
}

features = ["time", *sensors, *history]
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
