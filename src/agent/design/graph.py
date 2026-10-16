"""Design subgraph: agent writes prepare.py -> run + validate -> fix (bounded) -> done."""

import asyncio

from langgraph.graph import END, START, StateGraph

from agent import llm, mock_llm, prompts
from agent.research.edits import extract_code
from agent.state import DesignState
from agent.tools import design_tools
from core.config import settings
from core.events import bus
from ml import api as ml_api


async def write_prepare(state: DesignState) -> dict:
    ws, run_id = state["ws"], state["run_id"]
    report = state.get("report")
    feedback = "\n".join(report.messages) if report and not report.ok else "(first attempt)"
    if state.get("error"):
        feedback = f"{state['error']}\n{prompts.FORMAT_REMINDER_PREPARE}"
    elif state.get("code") and report and not report.ok:
        feedback += f"\n\nPrevious prepare.py:\n```python\n{state['code']}```"
    bus.emit(run_id, "design_attempt", task_id=ws.task_id, attempt=state.get("attempts", 0) + 1)
    reply = await llm.text(
        llm.messages(prompts.PREPARE_SYSTEM, prompts.PREPARE_USER.format(spec=ws.spec.model_dump_json(indent=2), feedback=feedback)),
        mock=lambda: mock_llm.prepare(ws.spec, feedback if report else ""),
        tools=design_tools(ws),
        # first attempt explores the raw streams; retries mostly fix validation messages
        max_tool_calls=settings.explore_tool_calls if not state.get("attempts") else settings.max_tool_calls,
        on_tool_call=lambda name, args: bus.emit(run_id, "tool_call", task_id=ws.task_id, tool=name, args=args),
    )
    code = extract_code(reply)
    if code is None:
        return {"error": "Your reply had no ```python block.", "code": ""}
    return {"code": code, "error": ""}


async def run_prepare(state: DesignState) -> dict:
    ws, run_id = state["ws"], state["run_id"]
    attempts = state.get("attempts", 0) + 1
    if not state.get("code"):
        return {"attempts": attempts}
    report = await asyncio.to_thread(ml_api.run_prepare, ws, state["code"])
    bus.emit(run_id, "validation", task_id=ws.task_id, attempt=attempts, ok=report.ok,
             checks=report.checks, messages=report.messages)
    return {"report": report, "attempts": attempts}


def route(state: DesignState) -> str:
    report = state.get("report")
    if report is not None and report.ok and state.get("code"):
        return END
    return "write_prepare" if state.get("attempts", 0) < settings.max_design_attempts else END


def build_design_graph():
    g = StateGraph(DesignState)
    g.add_node("write_prepare", write_prepare)
    g.add_node("run_prepare", run_prepare)
    g.add_edge(START, "write_prepare")
    g.add_edge("write_prepare", "run_prepare")
    g.add_conditional_edges("run_prepare", route, ["write_prepare", END])
    return g.compile()


design_graph = build_design_graph()
