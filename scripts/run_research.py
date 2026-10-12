"""Design + research loop on ONE hand-written TaskSpec (no formulate/discover/API).
The ML teammate's main loop for tuning harness, validators, program.md and recipes.

    uv run python scripts/run_research.py --spec specs/rul_example.json --n 10
    uv run python scripts/run_research.py --spec specs/rul_example.json --prepare my_prepare.py   # skip LLM design
"""

import argparse
import asyncio
from pathlib import Path

from agent.design.graph import design_graph
from agent.graph import new_run_id
from agent.research.graph import run_research
from core.config import settings
from core.contracts import TaskSpec
from core.events import bus
from ml import api as ml_api


async def printer(run_id: str) -> None:
    q = bus.subscribe(run_id)
    while True:
        e = await q.get()
        if e["type"] == "experiment":
            score = f"{e['score']:.4f}" if e.get("score") is not None else "-"
            print(f"{e['exp_id']} {e['status']:>8} score={score} {e['description']}")
        elif e["type"] in ("validation", "research_done", "proposal_invalid", "fix"):
            print({k: v for k, v in e.items() if k not in ("ts", "run_id", "best")})


async def main(spec_path: str, n: int, prepare_path: str | None) -> None:
    spec = TaskSpec.model_validate_json(Path(spec_path).read_text())
    settings.max_experiments = n
    run_id = new_run_id()
    bus.bind_loop(asyncio.get_running_loop())
    printer_task = asyncio.create_task(printer(run_id))

    ws = ml_api.create_workspace(run_id, spec)
    if prepare_path:
        report = await asyncio.to_thread(ml_api.run_prepare, ws, Path(prepare_path).read_text())
    else:
        report = (await design_graph.ainvoke({"run_id": run_id, "ws": ws, "attempts": 0})).get("report")
    if report is None or not report.ok:
        print("design failed:", report.messages if report else "no prepare.py")
        return
    ws = ml_api.freeze(ws)
    out = await run_research(run_id, ws)
    await asyncio.sleep(0.1)
    printer_task.cancel()
    print(f"\nstop: {out.get('stop_reason')}  best: {out['best'].exp_id if out.get('best') else None}")
    print(f"workspace: {ws.path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--n", type=int, default=10, help="max experiments")
    p.add_argument("--prepare", help="use this prepare.py instead of having the agent write one")
    args = p.parse_args()
    asyncio.run(main(args.spec, args.n, args.prepare))
