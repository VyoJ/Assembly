"""Research subgraph (the autoresearch loop):
explore (notes + plan) -> propose -> run -> [fix -> run]* -> check_stop -> [replan (every N)] -> propose ... -> END
"""

import asyncio
import difflib
import json
import re
import time
from pathlib import Path

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


def _history(ws, n: int) -> str:
    """Every experiment with its hypothesis, so repeats are visible (results.tsv only has short descriptions)."""
    rows = ws_files.list_experiments(Path(ws.path))[-n:]
    if not rows:
        return "(no experiments yet)"
    out = []
    for r in rows:
        primary = r["metrics"].get(ws.spec.primary_metric)
        kpis = f"{sum(r['kpis_passed'].values())}/{len(ws.spec.kpis)}"
        prim = f"{ws.spec.primary_metric}={primary:.4g}" if primary is not None else f"{ws.spec.primary_metric}=-"
        out.append(f"- {r['exp_id']} [{r['status']}] {prim} kpis={kpis} | {r['description']}\n"
                   f"  hypothesis: {r['hypothesis']}")
    return "\n".join(out)


def _tracer(ws, step: str):
    return lambda d: ws_files.log_llm_call(ws, {"step": step, **d})


def _tool_event(run_id: str, ws):
    return lambda name, args: bus.emit(run_id, "tool_call", task_id=ws.task_id, tool=name, args=args)


def _clean_notes(text: str) -> str:
    """Keep the prose, drop code and any experiment header the model wrote despite the instructions."""
    text = re.sub(r"```.*?(```|$)", "", text, flags=re.DOTALL)
    lines = [ln for ln in text.splitlines()
             if not re.match(r"^\s*\**(HYPOTHESIS|DESCRIPTION)\**\s*:", ln, re.IGNORECASE)]
    return "\n".join(lines).strip()


def _split_notes_plan(text: str) -> tuple[str, str]:
    """'## Notes ... ## Plan ...' -> (notes, plan), each cleaned; a missing section comes back empty."""
    text = _clean_notes(text)
    m = re.search(r"^#+\s*(research\s+)?plan\b.*$", text, re.MULTILINE | re.IGNORECASE)
    notes, plan = (text[:m.start()], text[m.end():]) if m else (text, "")
    notes = re.sub(r"^#+\s*(exploration\s+)?notes\b.*$", "", notes, count=1, flags=re.MULTILINE | re.IGNORECASE)
    return notes.strip(), plan.strip()


def _doc(ws, name: str) -> str:
    return ws_files.read_text(ws, name) or "(none)"


async def explore(state: ResearchState) -> dict:
    """Once per task, before exp_000: look at the actual data with tools, then write notes.md (verified facts,
    pitfalls) and plan.md (ranked experiments) from the same reply. Both go into every later step. Skipped
    when both already exist (e.g. a resumed task)."""
    ws, run_id = state["ws"], state["run_id"]
    if ws_files.read_text(ws, "notes.md") is not None and ws_files.read_text(ws, "plan.md") is not None:
        return {}
    msgs = llm.messages(
        ml_api.read_workspace_file(ws, "program.md") + "\n\n" + prompts.EXPLORE_SYSTEM,
        prompts.EXPLORE_USER.format(spec=ws.spec.model_dump_json(indent=2),
                                    data_card=ml_api.read_workspace_file(ws, "data_card.md")),
    )
    raw = await llm.text(
        msgs, mock=mock_llm.explore, tools=research_tools(ws), max_tool_calls=settings.explore_tool_calls,
        on_tool_call=_tool_event(run_id, ws), final_instruction=prompts.EXPLORE_FINAL, trace=_tracer(ws, "explore"),
    )
    notes, plan = _split_notes_plan(raw)
    # a missing section gets one follow-up without tools (e.g. the model answered with code or only one section)
    for missing, ask, step in ((not notes, prompts.NOTES_ONLY, "notes"), (not plan, prompts.PLAN_ONLY, "plan")):
        if missing:
            extra = await llm.followup(msgs, raw, ask, mock=mock_llm.explore)
            ws_files.log_llm_call(ws, {"step": f"explore_retry_{step}", "reply": extra, "tool_calls": []})
            got_notes, got_plan = _split_notes_plan(extra)
            if step == "notes":
                notes = got_notes or got_plan
            else:
                plan = got_plan or got_notes
    notes, plan = notes or "(exploration returned no notes)", plan or "(no plan)"
    ws_files.write_text(ws, "notes.md", notes)
    ws_files.write_text(ws, "plan.md", plan)
    bus.emit(run_id, "exploration", task_id=ws.task_id, notes=notes, raw=raw[:4000])
    bus.emit(run_id, "plan", task_id=ws.task_id, plan=plan, revision=False)
    return {"plan_at": 0}


async def replan(state: ResearchState) -> dict:
    """Revise plan.md from the results so far (every settings.plan_revise_every experiments). Read-only tools,
    so the model can look something up instead of faking a tool call in its text."""
    ws, run_id = state["ws"], state["run_id"]
    user = prompts.PLAN_USER.format(
        spec=ws.spec.model_dump_json(indent=2), data_card=ml_api.read_workspace_file(ws, "data_card.md"),
        notes=_doc(ws, "notes.md"), plan=_doc(ws, "plan.md"), history=_history(ws, settings.results_context_rows),
        best=_summarize(state.get("best")),
    )
    text = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md") + "\n\n" + prompts.REPLAN_SYSTEM, user),
        mock=mock_llm.plan, tools=research_tools(ws, read_only=True), max_tool_calls=2,
        on_tool_call=_tool_event(run_id, ws), final_instruction="Reply now with the revised plan only.",
        trace=_tracer(ws, "replan"),
    )
    _, plan = _split_notes_plan(text)
    plan = plan or _clean_notes(text) or _doc(ws, "plan.md")  # keep the old plan rather than write nothing
    ws_files.write_text(ws, "plan.md", plan)
    bus.emit(run_id, "plan", task_id=ws.task_id, plan=plan, revision=True)
    return {"plan_at": state.get("n_exp", 0), "replan_due": False}


def _norm_code(code: str) -> str:
    """Code without comments, blank lines and trailing / repeated whitespace."""
    lines = (re.sub(r"\s+", " ", ln.split("#")[0]).strip() for ln in code.splitlines())
    return "\n".join(ln for ln in lines if ln)


def _duplicate_of(ws, code: str) -> str | None:
    """exp_id whose train.py is the same code as `code` (comments and whitespace ignored). A changed value is a
    different experiment, so only exact repeats are rejected."""
    target = _norm_code(code)
    for d in sorted(Path(ws.path).glob("exp_*")):
        p = d / "train.py"
        if p.is_file() and _norm_code(p.read_text()) == target:
            return d.name
    return None


async def propose(state: ResearchState) -> dict:
    ws, run_id = state["ws"], state["run_id"]
    best_code = ws_files.best_train_py(ws)
    exp_id = ws_files.next_exp_id(ws)
    user = prompts.RESEARCH_USER.format(
        spec=ws.spec.model_dump_json(indent=2),
        data_card=ml_api.read_workspace_file(ws, "data_card.md"),
        notes=_doc(ws, "notes.md"),
        plan=_doc(ws, "plan.md"),
        recipes=(RECIPES_DIR / "INDEX.md").read_text(),
        best_code=f"```python\n{best_code}```" if best_code else "(none yet: write experiment 0 from scratch as a full ```python block)",
        history=_history(ws, settings.results_context_rows),
        last=_summarize(state.get("last")),
        feedback=f"# Problem with your previous answer\n{state['feedback']}\n" if state.get("feedback") else "",
        exp_id=exp_id,
    )
    reply = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md"), user),
        mock=lambda: mock_llm.propose(best_code, exp_id),
        tools=research_tools(ws),
        tool_limits={"run_python": settings.propose_run_python_calls},
        on_tool_call=_tool_event(run_id, ws),
        final_instruction=prompts.FORMAT_REMINDER,
        trace=_tracer(ws, f"propose {exp_id}"),
    )
    try:
        proposal = parse_proposal(reply)
        code = apply_edits(best_code, proposal.edits)
        check_train_py(code)
        dup = _duplicate_of(ws, code)
        if dup:
            raise EditError(f"This train.py is the same code as {dup}, which was already run. Try a different "
                            "idea (see the history and plan.md).")
    except EditError as e:
        bus.emit(run_id, "proposal_invalid", task_id=ws.task_id, error=str(e))
        return {"pending_code": None, "feedback": f"{e}\n{prompts.FORMAT_REMINDER}",
                "bad_proposals": state.get("bad_proposals", 0) + 1}
    bus.emit(run_id, "proposal", task_id=ws.task_id, exp_id=exp_id,
             hypothesis=proposal.hypothesis, description=proposal.description)
    return {"pending_code": code, "feedback": "", "fix_attempts": 0, "fix_history": [],
            "pending_meta": {"hypothesis": proposal.hypothesis, "description": proposal.description}}


async def run(state: ResearchState) -> dict:
    ws, run_id = state["ws"], state["run_id"]

    def progress(p: dict) -> None:
        bus.emit(run_id, "progress", task_id=ws.task_id, **p)

    result = await asyncio.to_thread(ml_api.run_experiment, ws, state["pending_code"], state["pending_meta"], progress)
    bus.emit(run_id, "experiment", **result.model_dump())
    return {"last": result, "best": ws_files.best_result(ws), "n_exp": state.get("n_exp", 0) + 1, "pending_code": None}


def _render_fix_history(ws, history: list[dict], current: str) -> str:
    """Earlier fixes in this chain: what each changed and how it ended, so a failed idea isn't repeated."""
    if not history:
        return "(none: this is the first fix attempt)"
    out = []
    for i, h in enumerate(history, 1):
        if h["runs_as"] == current:
            outcome = "failed: that is the failure shown above"
        else:
            r = ws_files.read_text(ws, f"{h['runs_as']}/result.json")
            tail = json.loads(r).get("log_tail", "")[-600:] if r else ""
            outcome = f"failed:\n{tail}"
        out.append(f"## Attempt {i}: {h['of']} -> {h['runs_as']}: {h['description']}\n```diff\n{h['diff']}```\n"
                   f"Outcome: {outcome}")
    return "\n\n".join(out)


async def fix(state: ResearchState) -> dict:
    """Same context as propose (program.md, task, data card, notes, tools) plus the failing code and its log, so a
    fix keeps to the harness API instead of guessing. An unusable reply counts as an attempt and is retried with
    the reason as feedback."""
    ws, last, run_id = state["ws"], state["last"], state["run_id"]
    failing = ws_files.read_file(ws, f"{last.exp_id}/train.py")
    history = state.get("fix_history", [])
    user = prompts.FIX_USER.format(
        previous=_render_fix_history(ws, history, last.exp_id),
        spec=ws.spec.model_dump_json(indent=2),
        data_card=ml_api.read_workspace_file(ws, "data_card.md"),
        notes=_doc(ws, "notes.md"),
        code=failing, status=last.status, log_tail=last.log_tail[-3000:],
        feedback=f"# Problem with your previous fix\n{state['feedback']}\n" if state.get("feedback") else "",
    )
    reply = await llm.text(
        llm.messages(ml_api.read_workspace_file(ws, "program.md") + "\n\n" + prompts.FIX_SYSTEM, user),
        mock=lambda: mock_llm.fix(failing),
        tools=research_tools(ws),
        max_tool_calls=settings.max_fix_tool_calls,
        on_tool_call=_tool_event(run_id, ws),
        final_instruction=prompts.FORMAT_REMINDER,
        trace=_tracer(ws, f"fix {last.exp_id}"),
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
    entry = {"of": last.exp_id, "runs_as": ws_files.next_exp_id(ws), "description": proposal.description,
             "diff": "".join(difflib.unified_diff(failing.splitlines(True), code.splitlines(True), n=1))[:3000]}
    return {"pending_code": code, "pending_meta": meta, "fix_attempts": attempts, "feedback": "",
            "fix_history": [*history, entry]}


async def check_stop(state: ResearchState) -> dict:
    reason = judge.stop_reason(state)
    if reason:
        bus.emit(state["run_id"], "research_done", task_id=state["ws"].task_id, reason=reason,
                 best=state["best"].model_dump() if state.get("best") else None, n_exp=state.get("n_exp", 0))
        return {"stop_reason": reason}
    due = state.get("n_exp", 0) - state.get("plan_at", 0) >= settings.plan_revise_every
    return {"stop_reason": None, "replan_due": due}


def after_propose(state: ResearchState) -> str:
    return "run" if state.get("pending_code") else "check_stop"


def after_run(state: ResearchState) -> str:
    return "fix" if judge.needs_fix(state) else "check_stop"


def after_fix(state: ResearchState) -> str:
    if state.get("pending_code"):
        return "run"
    return "fix" if judge.needs_fix(state) else "check_stop"


def after_check(state: ResearchState) -> str:
    if state.get("stop_reason"):
        return END
    return "replan" if state.get("replan_due") else "propose"


def build_research_graph():
    g = StateGraph(ResearchState)
    for name, fn in [("explore", explore), ("replan", replan), ("propose", propose), ("run", run), ("fix", fix),
                     ("check_stop", check_stop)]:
        g.add_node(name, fn)
    g.add_edge(START, "explore")
    g.add_edge("explore", "propose")
    g.add_edge("replan", "propose")
    g.add_conditional_edges("propose", after_propose, ["run", "check_stop"])
    g.add_conditional_edges("run", after_run, ["fix", "check_stop"])
    g.add_conditional_edges("fix", after_fix, ["run", "fix", "check_stop"])
    g.add_conditional_edges("check_stop", after_check, ["replan", "propose", END])
    return g.compile()


research_graph = build_research_graph()


async def run_research(run_id: str, ws) -> ResearchState:
    init: ResearchState = {"run_id": run_id, "ws": ws, "started_at": time.time(), "n_exp": 0,
                           "best": ws_files.best_result(ws), "last": None}
    return await research_graph.ainvoke(init, config={"recursion_limit": 10_000})
