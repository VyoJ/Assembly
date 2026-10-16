from typing import TypedDict

from core.contracts import (
    Deployment,
    ExperimentResult,
    FinalReport,
    LabelSource,
    ProblemSpec,
    StreamInfo,
    StreamProfile,
    ValidationReport,
    Workspace,
)


class OuterState(TypedDict, total=False):
    run_id: str
    goal: str
    auto_approve: bool
    status: str  # running | awaiting_approval | rejected | no_tasks | done
    spec: ProblemSpec
    streams: list[StreamInfo]
    label_sources: list[LabelSource]
    profiles: dict[str, list[StreamProfile]]
    workspaces: dict[str, Workspace]
    dropped: dict[str, str]  # task_id -> reason
    best: dict[str, ExperimentResult]
    full: dict[str, ExperimentResult]
    reports: dict[str, FinalReport]
    summary: str
    deployment: Deployment


class DesignState(TypedDict, total=False):
    run_id: str
    ws: Workspace
    code: str
    report: ValidationReport
    attempts: int
    error: str


class ResearchState(TypedDict, total=False):
    run_id: str
    ws: Workspace
    started_at: float
    n_exp: int
    best: ExperimentResult | None
    last: ExperimentResult | None
    pending_code: str | None
    pending_meta: dict
    fix_attempts: int
    bad_proposals: int  # unusable replies in a row
    rejected_proposals: int  # failed pre-run checks (after repairs) / exact repeats, in a row
    feedback: str
    stop_reason: str | None
    plan_at: int  # n_exp when plan.md was last written
    replan_due: bool
    fix_history: list[dict]  # fixes in the current chain: {of, runs_as, description, diff}
