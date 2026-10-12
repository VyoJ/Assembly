"""Shared schemas: the agent <-> ML interface. Change only after both teammates agree."""

from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["video", "image", "vibration", "pressure", "temperature", "plc", "log", "other"]

# The ONLY fixed vocabulary: how predictions look, so the fixed scorer can grade them.
PredictionFormat = Literal[
    "scalar",  # one float per sample (regression, RUL, ...)
    "binary_score",  # one score per sample + threshold (failure in N cycles, anomaly, defect present...)
    "class_probs",  # probability vector per sample (failure mode, ...)
    "boxes",  # per image: [x1, y1, x2, y2, score, class] list
    "masks",  # per image: instance masks + scores
]


class StreamInfo(BaseModel):
    id: str
    name: str
    modality: Modality
    unit: str | None = None
    source: str
    sample_hz: float | None = None
    description: str = ""


class LabelSource(BaseModel):
    id: str
    name: str
    description: str
    kind: Literal["events", "annotations", "table"]


class KPI(BaseModel):
    metric: str = Field(description="Must exist in the metric registry for the task's prediction format")
    op: Literal["<=", ">="]
    threshold: float


class TaskSpec(BaseModel):
    """Written by the agent, approved by the human, then frozen."""

    id: str = Field(description="short snake_case id, e.g. press_failure_30c")
    name: str
    rationale: str = Field(description="why this task serves the business goal")
    framing: str = Field(description="free text, e.g. 'predict whether an engine fails within 30 cycles'")
    input_streams: list[str]
    label_sources: list[str]
    target_definition: str = Field(description="free text, implemented by prepare.py")
    prediction_format: PredictionFormat
    primary_metric: str
    metrics: list[str]
    kpis: list[KPI]
    split_strategy: str = Field(description="e.g. 'by engine unit', 'by time', 'stratified by defect type'")
    time_budget_s: int = Field(description="wall-clock training budget per experiment, in seconds")


class ProblemSpec(BaseModel):
    goal: str
    tasks: list[TaskSpec]


class StreamProfile(BaseModel):
    stream_id: str
    relevance: float
    keep: bool
    reason: str


class ValidationReport(BaseModel):
    ok: bool
    checks: dict[str, bool] = {}
    messages: list[str] = []
    data_card: str = ""
    log_tail: str = ""


class Workspace(BaseModel):
    run_id: str
    task_id: str
    path: str
    spec: TaskSpec
    frozen: bool = False


class Proposal(BaseModel):
    hypothesis: str
    description: str
    edits: list[dict] = Field(description='[{"old": ..., "new": ...}] or [{"rewrite": "<full file>"}]')


ExperimentStatus = Literal["keep", "discard", "crash", "timeout", "invalid_output"]


class ExperimentResult(BaseModel):
    exp_id: str
    task_id: str
    status: ExperimentStatus
    score: float | None = None  # lower is better
    metrics: dict[str, float] = {}  # validation metrics only
    kpis_passed: dict[str, bool] = {}
    train_seconds: float = 0.0
    peak_mem_gb: float | None = None
    description: str = ""
    hypothesis: str = ""
    log_tail: str = ""
    diff: str = ""

    @property
    def all_kpis_passed(self) -> bool:
        return bool(self.kpis_passed) and all(self.kpis_passed.values())


class FinalReport(BaseModel):
    task_id: str
    exp_id: str
    test_metrics: dict[str, float]
    passed: bool
    failed_kpis: list[str]


class Deployment(BaseModel):
    id: str
    run_id: str
    task_ids: list[str]
    exp_ids: list[str]
    package_paths: dict[str, str]
    endpoint: str
    status: Literal["live", "stopped"] = "live"
