"""Hand-written prepare.py for specs/bearing_fault_example.json: raw waveforms -> fixed-size windows.

Shows the signal + array path: recordings are loaded with harness.load_signal, cut into windows and
written as an N-d array. Split by motor load, so val/test are operating conditions never seen in training.

    uv run python scripts/run_research.py --spec specs/bearing_fault_example.json --prepare specs/bearing_fault_example_prepare.py
"""

import harness
import numpy as np
from scipy.signal import decimate

TARGET_HZ = 12_000
WINDOW = 2048  # ~0.17 s at 12 kHz
STRIDE = 1024
SPLIT_BY_LOAD = {"train": [0, 1], "val": [2], "test": [3]}

spec = harness.task()
recordings = harness.read_stream(spec["input_streams"][0])
labels = harness.read_labels("bearing_inspections")
recordings = recordings.merge(labels[["record_id", "fault"]], on="record_id")

parts = {}
for split, loads in SPLIT_BY_LOAD.items():
    X, y, groups = [], [], []
    for _, rec in recordings[recordings["load_hp"].isin(loads)].iterrows():
        signal = harness.load_signal(rec["path"], rec["channel"])
        q = int(rec["sample_hz"]) // TARGET_HZ
        if q > 1:  # 48 kHz normal records -> 12 kHz, low-pass first so high frequencies don't alias
            signal = decimate(signal, q, ftype="fir", zero_phase=True)
        for start in range(0, len(signal) - WINDOW + 1, STRIDE):
            X.append(signal[start:start + WINDOW])
            y.append(rec["fault"])
            groups.append(rec["record_id"])
    parts[split] = (np.stack(X).astype("float32"), y, groups)

harness.write_dataset(
    parts["train"][:2], parts["val"][:2], parts["test"][:2],
    groups={s: p[2] for s, p in parts.items()},
    meta={"split": "group", "group_col": "load_hp", "sample_hz": TARGET_HZ, "window": WINDOW,
          "normal_class": "normal"},
)
