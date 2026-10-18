"""formulate (goal -> task specs) and discover (which streams matter)."""

import asyncio
import re

from langchain_core.messages import AIMessage, HumanMessage

from agent import llm, mock_llm, prompts
from agent.state import OuterState
from core.config import settings
from core.contracts import LabelSource, ProblemSpec, StreamInfo, TaskSpec
from core.events import bus
from ml import api as ml_api
from ml import scoring
from ml.workspace import installed_packages


def _problems(spec: ProblemSpec, streams: list[StreamInfo], labels: list[LabelSource]) -> list[str]:
    stream_ids, label_ids = {s.id for s in streams}, {lab.id for lab in labels}
    out = []
    if not spec.tasks:
        out.append("no tasks were proposed")
    for t in spec.tasks:
        out += [f"{t.id}: {p}" for p in scoring.validate_spec_metrics(t)]
        out += [f"{t.id}: unknown stream '{s}'" for s in t.input_streams if s not in stream_ids]
        out += [f"{t.id}: unknown label source '{s}'" for s in t.label_sources if s not in label_ids]
        if not t.kpis:
            out.append(f"{t.id}: no KPIs")
    return out


def _sanitize(spec: ProblemSpec, streams: list[StreamInfo], labels: list[LabelSource]) -> ProblemSpec:
    """Best-effort repair after the retry: drop unknown ids/metrics, clamp budgets, unique ids."""
    stream_ids, label_ids = {s.id for s in streams}, {lab.id for lab in labels}
    tasks, seen = [], set()
    for t in spec.tasks:
        registry = scoring.METRIC_REGISTRY[t.prediction_format]
        tid = re.sub(r"[^a-z0-9_]+", "_", t.id.lower()).strip("_") or "task"
        while tid in seen:
            tid += "_2"
        seen.add(tid)
        metrics = [m for m in t.metrics if m in registry] or [next(iter(registry))]
        primary = t.primary_metric if t.primary_metric in registry else metrics[0]
        tasks.append(t.model_copy(update={
            "id": tid,
            "input_streams": [s for s in t.input_streams if s in stream_ids],
            "label_sources": [s for s in t.label_sources if s in label_ids],
            "metrics": list(dict.fromkeys([primary, *metrics])),
            "primary_metric": primary,
            "kpis": [k for k in t.kpis if k.metric in registry],
            "time_budget_s": settings.clamp_budget(t.time_budget_s),
        }))
    return spec.model_copy(update={"tasks": tasks})


async def formulate(state: OuterState) -> dict:
    run_id, goal = state["run_id"], state["goal"]
    bus.emit(run_id, "stage", stage="formulate")
    streams, labels = ml_api.list_catalog()
    msgs = llm.messages(
        prompts.FORMULATE_SYSTEM.format(min_budget=settings.min_budget_s, max_budget=settings.max_budget_s),
        prompts.FORMULATE_USER.format(
            goal=goal,
            streams="\n".join(f"- {s.id} [{s.modality}] {s.name}. {s.description}" for s in streams),
            label_sources="\n".join(f"- {lab.id} [{lab.kind}] {lab.description}" for lab in labels),
            registry=llm.dumps(ml_api.metric_registry()),
            packages=", ".join(installed_packages()) or "(ML group not installed here)",
        ),
    )
    mock = lambda: mock_llm.formulate(goal, [s.id for s in streams])
    spec = await llm.structured(ProblemSpec, msgs, mock=mock)
    problems = _problems(spec, streams, labels)
    if problems:
        msgs += [AIMessage(spec.model_dump_json()), HumanMessage(prompts.FORMULATE_FIX.format(problems="\n".join(problems)))]
        spec = await llm.structured(ProblemSpec, msgs, mock=mock)
        problems = _problems(spec, streams, labels)
    spec = _sanitize(spec.model_copy(update={"goal": goal}), streams, labels)
    bus.emit(run_id, "task_specs", tasks=[t.model_dump() for t in spec.tasks], warnings=problems)
    return {"spec": spec, "streams": streams, "label_sources": labels, "status": "running"}


async def discover(state: OuterState) -> dict:
    run_id, spec, streams = state["run_id"], state["spec"], state["streams"]
    bus.emit(run_id, "stage", stage="discover")
    by_id = {s.id: s for s in streams}

    async def one(t: TaskSpec) -> tuple[TaskSpec, list]:
        # only streams of the same data kind as the formulated ones: a sensor-table task must not pick up another
        # machine's recordings (they can't be joined on entity x time) or camera images, and vice versa
        kinds = {ml_api.stream_kind(s) for s in t.input_streams if s in by_id}
        candidates = [s for s in streams if ml_api.stream_kind(s.id) in kinds] if kinds else list(streams)
        profiles = await asyncio.to_thread(ml_api.profile_streams, t, candidates)
        kept = [p.stream_id for p in profiles if p.keep]
        bus.emit(run_id, "discovery", task_id=t.id, scanned=len(candidates), kept=len(kept),
                 profiles=[p.model_dump() for p in profiles])
        return t.model_copy(update={"input_streams": kept or t.input_streams}), profiles

    results = await asyncio.gather(*(one(t) for t in spec.tasks))
    return {
        "spec": spec.model_copy(update={"tasks": [t for t, _ in results]}),
        "profiles": {t.id: p for t, p in results},
    }
