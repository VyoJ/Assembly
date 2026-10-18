"""Outer pipeline:
formulate -> discover -> design -> approve_plan -> research -> full_train -> final_gate -> deploy
"""

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from agent.nodes.build import approve_plan, design, research
from agent.nodes.plan import discover, formulate
from agent.nodes.ship import deploy, final_gate, full_train
from agent.state import OuterState
from core.events import bus


def _continue_if_running(next_node: str):
    def route(state: OuterState) -> str:
        return next_node if state.get("status") == "running" else END

    return route


def build_graph():
    g = StateGraph(OuterState)
    for name, fn in [("formulate", formulate), ("discover", discover), ("design", design),
                     ("approve_plan", approve_plan), ("research", research), ("full_train", full_train),
                     ("final_gate", final_gate), ("deploy", deploy)]:
        g.add_node(name, fn)
    g.add_edge(START, "formulate")
    g.add_edge("formulate", "discover")
    g.add_edge("discover", "design")
    g.add_conditional_edges("design", _continue_if_running("approve_plan"), ["approve_plan", END])
    g.add_conditional_edges("approve_plan", _continue_if_running("research"), ["research", END])
    g.add_edge("research", "full_train")
    g.add_edge("full_train", "final_gate")
    g.add_edge("final_gate", "deploy")
    g.add_edge("deploy", END)
    return g.compile()


graph = build_graph()


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


async def run_pipeline(
    goal: str,
    run_id: str | None = None,
    auto_approve: bool = False,
    on_state: Callable[[OuterState], None] | None = None,
) -> OuterState:
    run_id = run_id or new_run_id()
    bus.bind_loop(asyncio.get_running_loop())
    bus.emit(run_id, "run_started", goal=goal, auto_approve=auto_approve)
    final: OuterState = {}
    try:
        init: OuterState = {"run_id": run_id, "goal": goal, "auto_approve": auto_approve, "status": "running"}
        async for state in graph.astream(init, stream_mode="values"):
            final = state
            if on_state:
                on_state(state)
    except Exception as e:
        bus.emit(run_id, "run_failed", error=repr(e))
        raise
    bus.emit(run_id, "run_finished", status=final.get("status"))
    return final
