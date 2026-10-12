"""Canned LLM responses for LLM_MOCK=1. They use the real output formats, so parsing is exercised."""

import random
import re

from core.contracts import KPI, ProblemSpec, TaskSpec

_rng = random.Random(11)


def formulate(goal: str, stream_ids: list[str]) -> ProblemSpec:
    sensors = [s for s in stream_ids if s.startswith("press3.")]
    return ProblemSpec(goal=goal, tasks=[
        TaskSpec(
            id="press_failure_30c", name="Press failure within 30 cycles",
            rationale="Early warning before a press fails lets maintenance act before the line stops.",
            framing="Binary: will this press unit fail within the next 30 cycles, from its sensor history?",
            input_streams=sensors, label_sources=["maintenance_log"],
            target_definition="y=1 if failure_cycle - current_cycle <= 30 else 0, per unit and cycle",
            prediction_format="binary_score", primary_metric="pr_auc",
            metrics=["pr_auc", "recall", "precision", "false_alarm_rate", "p95_latency_ms"],
            kpis=[KPI(metric="recall", op=">=", threshold=0.9), KPI(metric="false_alarm_rate", op="<=", threshold=0.05)],
            split_strategy="by press unit (70/15/15)", time_budget_s=180,
        ),
        TaskSpec(
            id="part_defects", name="Defective part detection",
            rationale="Catching defective parts at the camera stops bad parts from jamming downstream stations.",
            framing="Detect surface defects on each part image from the top camera.",
            input_streams=["line3.cam_top"], label_sources=["qc_inspections"],
            target_definition="boxes derived from QC defect masks; images with no defect have no boxes",
            prediction_format="boxes", primary_metric="map50",
            metrics=["map50", "image_recall", "image_far", "p95_latency_ms"],
            kpis=[KPI(metric="image_recall", op=">=", threshold=0.9), KPI(metric="image_far", op="<=", threshold=0.05),
                  KPI(metric="p95_latency_ms", op="<=", threshold=50)],
            split_strategy="stratified by defect type, by image (70/15/15)", time_budget_s=600,
        ),
    ])


def prepare(spec: TaskSpec, feedback: str) -> str:
    split = "GroupShuffleSplit by unit" if "unit" in spec.split_strategy else "stratified split"
    return f'''Here is prepare.py:
```python
"""prepare.py for {spec.id} (mock). {"Fixed: " + feedback[:60] if feedback else ""}"""
import harness

# Build X / y per target_definition: {spec.target_definition}
# Split: {split}
streams = {{sid: harness.read_stream(sid) for sid in {spec.input_streams[:3]!r}}}
labels = harness.read_labels({spec.label_sources[0]!r})
# ... feature assembly elided in mock ...
harness.write_dataset(train=(None, None), val=(None, None), test=(None, None), groups={{}})
```'''


def _template(model: str, lr: float, window: int) -> str:
    return f'''import harness

MODEL = "{model}"
LEARNING_RATE = {lr}
WINDOW = {window}
N_ESTIMATORS = 400


def main():
    X, y = harness.load_train()
    # ... build {model} with window=WINDOW, train until harness.time_left() < 10 ...
    model = None
    harness.save_model(model)
    harness.finish(lambda X: [0.5] * len(X))


if __name__ == "__main__":
    main()
'''


_IDEAS = [
    ("LEARNING_RATE", "0.05", "lower learning rate should reduce overfitting on few failures", "lower learning rate"),
    ("WINDOW", "50", "longer windows capture the slow degradation trend", "window 30 -> 50"),
    ("MODEL", '"lstm"', "an LSTM over the window can model temporal dynamics trees miss", "switch to LSTM"),
    ("N_ESTIMATORS", "800", "more trees with early stopping", "more trees"),
    ("LEARNING_RATE", "0.02", "even lower LR with longer training", "LR 0.02"),
]


def propose(best_code: str | None, exp_id: str) -> str:
    if not best_code:
        return ("HYPOTHESIS: A simple gradient-boosted baseline on window features gives a valid first score.\n"
                "DESCRIPTION: xgboost baseline on window features\n"
                f"```python\n{_template('xgboost', 0.1, 30)}```")
    var, value, hyp, desc = _IDEAS[int(exp_id.split('_')[1]) % len(_IDEAS)]
    m = re.search(rf"^{var} = .*$", best_code, re.MULTILINE)
    if not m:
        return f"HYPOTHESIS: {hyp}\nDESCRIPTION: {desc}\n```python\n{_template('xgboost', 0.07, 40)}```"
    return (f"HYPOTHESIS: {hyp}\nDESCRIPTION: {desc}\n"
            f"<<<<<<< SEARCH\n{m.group(0)}\n=======\n{var} = {value}\n>>>>>>> REPLACE")


def fix(code: str) -> str:
    return ("HYPOTHESIS: Feature count mismatch; recompute features after dropping constant columns.\n"
            "DESCRIPTION: fix feature shape mismatch\n"
            f"```python\n# fixed: align feature columns\n{code}```")


def report(goal: str, results: str) -> str:
    return (f"Goal: {goal}\nAssembly trained and deployed models for each task below. "
            f"Results on held-out test data:\n{results}\n(mock report)")
