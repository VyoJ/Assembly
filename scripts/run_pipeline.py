"""Run the whole agent pipeline headless and print events.

    ML_MOCK=1 LLM_MOCK=1 uv run python scripts/run_pipeline.py --goal "Prevent line stoppage on Line 3"
"""

import argparse
import asyncio
import json

from agent.graph import new_run_id, run_pipeline
from core.events import bus

QUIET = {"progress", "tool_call"}


def show(e: dict, verbose: bool) -> None:
    t = e["type"]
    if t in QUIET and not verbose:
        return
    if t == "stage":
        print(f"\n=== {e['stage']} ===")
    elif t == "experiment":
        kpis = sum(e["kpis_passed"].values()) if e.get("kpis_passed") else 0
        score = f"{e['score']:.4f}" if e.get("score") is not None else "-"
        print(f"  [{e['task_id']}] {e['exp_id']} {e['status']:>8}  score={score}  kpis={kpis}/{len(e.get('kpis_passed') or {})}  {e['description']}")
    elif t == "task_specs":
        for task in e["tasks"]:
            print(f"  task {task['id']}: {task['framing']} [{task['prediction_format']}] kpis={[(k['metric'], k['op'], k['threshold']) for k in task['kpis']]}")
    elif t == "discovery":
        print(f"  [{e['task_id']}] scanned {e['scanned']} streams, kept {e['kept']}")
    elif t == "validation":
        print(f"  [{e['task_id']}] prepare.py attempt {e['attempt']}: {'OK' if e['ok'] else 'FAILED ' + '; '.join(e['messages'])}")
    elif t == "report":
        print("\n" + e["summary"])
    elif t in ("proposal", "design_attempt", "approved", "run_started"):
        if verbose:
            print(f"  {t}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 'run_id', 'type')})[:200]}")
    else:
        print(f"  {t}: {json.dumps({k: v for k, v in e.items() if k not in ('ts', 'run_id', 'type')}, default=str)[:300]}")


async def main(goal: str, auto_approve: bool, verbose: bool) -> None:
    run_id = new_run_id()
    q = bus.subscribe(run_id)
    task = asyncio.create_task(run_pipeline(goal, run_id=run_id, auto_approve=auto_approve))
    while not (task.done() and q.empty()):
        try:
            show(await asyncio.wait_for(q.get(), timeout=0.5), verbose)
        except TimeoutError:
            continue
    await task
    print(f"\nrun_id={run_id}  (workspaces + events.jsonl in runs/{run_id}/)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="Prevent unplanned stoppages on Press Line 3. Max 1 false alarm per shift; catch at least 90% of failures.")
    p.add_argument("--no-auto-approve", action="store_true", help="wait for POST /approve (only useful via the API)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args.goal, not args.no_auto_approve, args.verbose))
