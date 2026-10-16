"""Research subgraph (the autoresearch loop):
propose -> run -> [fix -> run]* -> check_stop -> propose ... -> END
"""

import asyncio
import re
import time

from langgraph.graph import END, START, StateGraph

from agent import llm, mock_llm, prompts
from agent.research import judge
from agent.research.edits import EditError, apply_edits, check_train_py, parse_proposal
from agent.state import ResearchState
from agent.tools import RECIPES_DIR, research_tools
from core.config import settings
from core.contracts import ExperimentResult
from core.events import bus
from ml import api as ml_api
from ml import workspace as ws_files


def _summarize(r: ExperimentResult | None) -> str:
    if r is None:
        return "(none yet)"
    metrics = ", ".join(f"{k}={v:.4g}" for k, v in r.metrics.items())
    return (f"{r.exp_id} [{r.status}] score={r.score} | {r.description}\nmetrics: {metrics or '-'}\n"
            f"kpis: {r.kpis_passed or '-'}\nlog tail:\n{r.log_tail[-1500:]}")


async def explore(state: ResearchState) -> dict:
    """Once per task, before exp_000: look at the actual data with tools and write notes.md, which goes into
    every later propose / fix. Skipped when notes.md already exists (e.g. a resumed task)."""
    ws, run_id = state["ws"], state["run_id"]
    if ws_files.read_notes(ws) is not None:
        return {}
    user = prompts.EXPLORE_USER.format(spec=ws.spec.model_dump_json(indent=2),
                                       data_card=ml_api.read_workspace_file(ws, "data_card.md"))
    notes = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md") + "\n\n" + prompts.EXPLORE_SYSTEM, user),
        mock=mock_llm.explore,
        tools=research_tools(ws),
        max_tool_calls=settings.explore_tool_calls,
        on_tool_call=lambda name, args: bus.emit(run_id, "tool_call", task_id=ws.task_id, tool=name, args=args),
    )
    notes = _clean_notes(notes)
    ws_files.write_notes(ws, notes)
    bus.emit(run_id, "exploration", task_id=ws.task_id, notes=notes)
    return {}


def _clean_notes(text: str) -> str:
    """Keep the findings, drop code and any experiment header the model wrote despite the instructions."""
    text = re.sub(r"```.*?(```|$)", "", text, flags=re.DOTALL)
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*\**(HYPOTHESIS|DESCRIPTION)\**\s*:", ln, re.IGNORECASE)]
    return "\n".join(lines).strip() or "(exploration returned no notes)"


def _notes(ws) -> str:
    return ws_files.read_notes(ws) or "(none)"


async def propose(state: ResearchState) -> dict:
    ws, run_id = state["ws"], state["run_id"]
    best_code = ws_files.best_train_py(ws)
    exp_id = ws_files.next_exp_id(ws)
    user = prompts.RESEARCH_USER.format(
        spec=ws.spec.model_dump_json(indent=2),
        data_card=ml_api.read_workspace_file(ws, "data_card.md"),
        notes=_notes(ws),
        recipes=(RECIPES_DIR / "INDEX.md").read_text(),
        best_code=f"```python\n{best_code}```" if best_code else "(none yet: write experiment 0 from scratch as a full ```python block)",
        results=ws_files.results_tail(ws, settings.results_context_rows),
        last=_summarize(state.get("last")),
        feedback=f"# Problem with your previous answer\n{state['feedback']}\n" if state.get("feedback") else "",
        exp_id=exp_id,
    )
    reply = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md"), user),
        mock=lambda: mock_llm.propose(best_code, exp_id),
        tools=research_tools(ws),
        on_tool_call=lambda name, args: bus.emit(run_id, "tool_call", task_id=ws.task_id, tool=name, args=args),
    )
    try:
        proposal = parse_proposal(reply)
        code = apply_edits(best_code, proposal.edits)
        check_train_py(code)
    except EditError as e:
        bus.emit(run_id, "proposal_invalid", task_id=ws.task_id, error=str(e))
        return {"pending_code": None, "feedback": f"{e}\n{prompts.FORMAT_REMINDER}",
                "bad_proposals": state.get("bad_proposals", 0) + 1}
    bus.emit(run_id, "proposal", task_id=ws.task_id, exp_id=exp_id,
             hypothesis=proposal.hypothesis, description=proposal.description)
    return {"pending_code": code, "feedback": "", "fix_attempts": 0,
            "pending_meta": {"hypothesis": proposal.hypothesis, "description": proposal.description}}


async def run(state: ResearchState) -> dict:
    ws, run_id = state["ws"], state["run_id"]

    def progress(p: dict) -> None:
        bus.emit(run_id, "progress", task_id=ws.task_id, **p)

    result = await asyncio.to_thread(ml_api.run_experiment, ws, state["pending_code"], state["pending_meta"], progress)
    bus.emit(run_id, "experiment", **result.model_dump())
    return {"last": result, "best": ws_files.best_result(ws), "n_exp": state.get("n_exp", 0) + 1, "pending_code": None}


async def fix(state: ResearchState) -> dict:
    """Same context as propose (program.md, task, data card, tools) plus the failing code and its log, so a fix
    keeps to the harness API instead of guessing. An unusable reply counts as an attempt and is retried with
    the reason as feedback."""
    ws, last, run_id = state["ws"], state["last"], state["run_id"]
    failing = ws_files.read_file(ws, f"{last.exp_id}/train.py")
    user = prompts.FIX_USER.format(
        spec=ws.spec.model_dump_json(indent=2),
        data_card=ml_api.read_workspace_file(ws, "data_card.md"),
        notes=_notes(ws),
        code=failing, status=last.status, log_tail=last.log_tail[-3000:],
        feedback=f"# Problem with your previous fix\n{state['feedback']}\n" if state.get("feedback") else "",
    )
    reply = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md") + "\n\n" + prompts.FIX_SYSTEM, user),
        mock=lambda: mock_llm.fix(failing),
        tools=research_tools(ws),
        max_tool_calls=settings.max_fix_tool_calls,
        on_tool_call=lambda name, args: bus.emit(run_id, "tool_call", task_id=ws.task_id, tool=name, args=args),
    )
    attempts = state.get("fix_attempts", 0) + 1
    try:
        proposal = parse_proposal(reply)
        code = apply_edits(failing, proposal.edits)
        check_train_py(code)
    except EditError as e:
        bus.emit(run_id, "proposal_invalid", task_id=ws.task_id, error=f"fix: {e}")
        return {"fix_attempts": attempts, "pending_code": None, "feedback": f"{e}\n{prompts.FORMAT_REMINDER}"}
    bus.emit(run_id, "fix", task_id=ws.task_id, of=last.exp_id, description=proposal.description)
    meta = {"hypothesis": last.hypothesis, "description": f"fix({last.exp_id}): {proposal.description}"}
    return {"pending_code": code, "pending_meta": meta, "fix_attempts": attempts, "feedback": ""}


async def check_stop(state: ResearchState) -> dict:
    reason = judge.stop_reason(state)
    if reason:
        bus.emit(state["run_id"], "research_done", task_id=state["ws"].task_id, reason=reason,
                 best=state["best"].model_dump() if state.get("best") else None, n_exp=state.get("n_exp", 0))
    return {"stop_reason": reason}


def after_propose(state: ResearchState) -> str:
    return "run" if state.get("pending_code") else "check_stop"


def after_run(state: ResearchState) -> str:
    return "fix" if judge.needs_fix(state) else "check_stop"


def after_fix(state: ResearchState) -> str:
    if state.get("pending_code"):
        return "run"
    return "fix" if judge.needs_fix(state) else "check_stop"


def after_check(state: ResearchState) -> str:
    return END if state.get("stop_reason") else "propose"


def build_research_graph():
    g = StateGraph(ResearchState)
    for name, fn in [("explore", explore), ("propose", propose), ("run", run), ("fix", fix), ("check_stop", check_stop)]:
        g.add_node(name, fn)
    g.add_edge(START, "explore")
    g.add_edge("explore", "propose")
    g.add_conditional_edges("propose", after_propose, ["run", "check_stop"])
    g.add_conditional_edges("run", after_run, ["fix", "check_stop"])
    g.add_conditional_edges("fix", after_fix, ["run", "fix", "check_stop"])
    g.add_conditional_edges("check_stop", after_check, ["propose", END])
    return g.compile()


research_graph = build_research_graph()


async def run_research(run_id: str, ws) -> ResearchState:
    init: ResearchState = {"run_id": run_id, "ws": ws, "started_at": time.time(), "n_exp": 0,
                           "best": ws_files.best_result(ws), "last": None}
    return await research_graph.ainvoke(init, config={"recursion_limit": 10_000})
