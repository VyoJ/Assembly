"""Deterministic stop rules for the research loop (no LLM)."""

import time

from agent.state import ResearchState
from core.config import settings
from ml import workspace as ws_files

FAILED = ("crash", "timeout", "invalid_output")
MAX_BAD_PROPOSALS = 4


def _experiments_since_best(state: ResearchState) -> int:
    best = state.get("best")
    total = int(ws_files.next_exp_id(state["ws"]).split("_")[1])
    if best is None or not best.exp_id.startswith("exp_"):
        return total
    return total - 1 - int(best.exp_id.split("_")[1])


def stop_reason(state: ResearchState) -> str | None:
    best = state.get("best")
    if best is not None and best.all_kpis_passed and _experiments_since_best(state) >= settings.extra_after_pass:
        return "kpis_met"
    if state.get("n_exp", 0) >= settings.max_experiments:
        return "max_experiments"
    if time.time() - state["started_at"] >= settings.research_budget_s:
        return "time_budget"
    if state.get("bad_proposals", 0) >= MAX_BAD_PROPOSALS:
        return "too_many_invalid_proposals"
    if best is not None and _experiments_since_best(state) >= settings.plateau_n:
        return "plateau"
    return None


def needs_fix(state: ResearchState) -> bool:
    last = state.get("last")
    return last is not None and last.status in FAILED and state.get("fix_attempts", 0) < settings.max_fix_attempts
