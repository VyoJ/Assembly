"""Runs agent-written code (prepare.py / train.py) in a subprocess and scores the result.

TODO(ML): implement. Bookkeeping (snapshots, keep/discard, results.tsv) already exists in
ml.workspace.record_experiment, so the runner only has to execute, time, and score.

Subprocess contract (env vars passed to the child; harness.py reads them):
    ASSEMBLY_WORKSPACE      workspace dir (also the cwd)
    ASSEMBLY_EXP_DIR        exp_NNN/ dir to write model/, predictions, metrics into
    ASSEMBLY_TIME_BUDGET_S  training budget (spec.time_budget_s, x multiplier for full_train)
    ASSEMBLY_WEIGHTS_DIR    settings.data_dir / "weights"
    ASSEMBLY_DATA_DIR       settings.data_dir (prepare.py only)
    ASSEMBLY_LABELS_DIR     hidden labels dir (prepare.py ONLY; never passed to train.py)
Do not pass LLM_* or other secrets. Hard-kill at 2x the budget -> status "timeout".
"""

from core.contracts import ExperimentResult, FinalReport, ValidationReport, Workspace


def run_prepare(ws: Workspace, prepare_py: str) -> ValidationReport:
    """TODO(ML):
    1. workspace.write_prepare(ws, prepare_py)
    2. run `python prepare.py` (cwd=workspace, timeout) with ASSEMBLY_LABELS_DIR set
    3. validators.validate(ws) -> ValidationReport (+ workspace.write_data_card on success)
    """
    raise NotImplementedError("TODO(ML): runner.run_prepare")


def run_experiment(ws: Workspace, train_py: str, meta: dict, progress=None) -> ExperimentResult:
    """TODO(ML):
    1. exp_id, d = workspace.new_exp_dir(ws, train_py)
    2. run `python train.py` (cwd=workspace, budget env, hard kill at 2x); stream stdout lines to progress()
    3. crash/timeout -> ExperimentResult(status=..., log_tail=last 40 lines)
    4. check predictions match spec.prediction_format, else status "invalid_output"
    5. metrics = scoring.compute_metrics(fmt, preds, hidden val labels, spec.metrics + kpi metrics + latency)
    6. score = scoring.compute_score(metrics, spec); kpis_passed = scoring.kpi_status(metrics, spec.kpis)
    7. return workspace.record_experiment(ws, result, train_py)
    """
    raise NotImplementedError("TODO(ML): runner.run_experiment")


def full_train(ws: Workspace, multiplier: float, progress=None) -> ExperimentResult:
    """TODO(ML): re-run best/train.py with budget x multiplier into full/; score on val (not recorded in results.tsv)."""
    raise NotImplementedError("TODO(ML): runner.full_train")


def final_test(ws: Workspace) -> FinalReport:
    """TODO(ML): score the best (or full-trained, if better on val) model's TEST predictions vs hidden test labels."""
    raise NotImplementedError("TODO(ML): runner.final_test")
