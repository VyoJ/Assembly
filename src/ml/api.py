"""PUBLIC FACADE: the only ML module the agent imports.

Each function dispatches to ml.mock when ML_MOCK=1, otherwise to the real implementation.
Real implementations marked TODO(ML) raise NotImplementedError until the ML teammate fills them in.
"""

from core.config import settings
from core.contracts import (
    ExperimentResult,
    FinalReport,
    LabelSource,
    StreamInfo,
    StreamProfile,
    TaskSpec,
    ValidationReport,
    Workspace,
)
from ml import catalog, mock, profiling, runner, scoring, workspace
from ml import package as packaging


def list_catalog() -> tuple[list[StreamInfo], list[LabelSource]]:
    return catalog.list_catalog()


def metric_registry() -> dict[str, dict[str, str]]:
    return scoring.registry_for_prompt()


def peek_stream(stream_id: str, n: int = 5) -> str:
    return mock.peek_stream(stream_id, n) if settings.ml_mock else catalog.peek_stream(stream_id, n)


def profile_streams(spec: TaskSpec, streams: list[StreamInfo]) -> list[StreamProfile]:
    return mock.profile_streams(spec, streams) if settings.ml_mock else profiling.profile_streams(spec, streams)


def create_workspace(run_id: str, spec: TaskSpec) -> Workspace:
    return workspace.create_workspace(run_id, spec)


def update_spec(ws: Workspace, spec: TaskSpec) -> Workspace:
    return workspace.update_spec(ws, spec)


def run_prepare(ws: Workspace, prepare_py: str) -> ValidationReport:
    """Write + run prepare.py, then validate its output. Repeatable until frozen."""
    return mock.run_prepare(ws, prepare_py) if settings.ml_mock else runner.run_prepare(ws, prepare_py)


def static_check(code: str) -> list[str]:
    """Pre-run check of agent code against the installed library signatures (see ml.static_check)."""
    return mock.static_check(code) if settings.ml_mock else runner.static_check(code)


def run_snippet(ws: Workspace, code: str, phase: str = "research") -> str:
    """Run a short read-only exploration script in the workspace sandbox; returns its capped output."""
    return mock.run_snippet(ws, code, phase=phase) if settings.ml_mock else runner.run_snippet(ws, code, phase=phase)


def freeze(ws: Workspace) -> Workspace:
    return workspace.freeze(ws)


def read_workspace_file(ws: Workspace, name: str) -> str:
    return workspace.read_file(ws, name)


def run_experiment(ws: Workspace, train_py: str, meta: dict, progress=None) -> ExperimentResult:
    """Snapshot to exp_NNN/, run under spec.time_budget_s, score vs hidden val labels, keep/discard."""
    if settings.ml_mock:
        return mock.run_experiment(ws, train_py, meta, progress)
    return runner.run_experiment(ws, train_py, meta, progress)


def full_train(ws: Workspace, multiplier: float | None = None, progress=None) -> ExperimentResult:
    multiplier = multiplier or settings.full_train_multiplier
    if settings.ml_mock:
        return mock.full_train(ws, multiplier, progress)
    return runner.full_train(ws, multiplier, progress)


def final_test(ws: Workspace) -> FinalReport:
    return mock.final_test(ws) if settings.ml_mock else runner.final_test(ws)


def package(ws: Workspace) -> str:
    return mock.package(ws) if settings.ml_mock else packaging.package(ws)


def load_predictor(package_path: str):
    return mock.load_predictor(package_path) if settings.ml_mock else packaging.load_predictor(package_path)
