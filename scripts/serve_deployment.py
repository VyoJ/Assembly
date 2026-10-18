"""Start the API with packaged winners from earlier runs already deployed (no pipeline run needed).

Useful as a demo fallback and for frontend work on the live feed: one deployment can combine tasks from several
runs (e.g. a sensor model and a camera model).

    uv run python scripts/serve_deployment.py runs/<run_a> runs/<run_b>/<task> --port 8000
    # -> GET /api/deployments shows "dep_served"; the frontend's Deployments page streams its live feed
"""

import argparse
from pathlib import Path

import uvicorn

from api.main import app
from core import registry
from core.contracts import Deployment, TaskSpec
from ml import api as ml_api


def packages(paths: list[str]) -> list[Path]:
    """Task workspaces with a package/, from run dirs (all tasks) or task dirs."""
    out = []
    for p in map(Path, paths):
        candidates = [p] if (p / "package").is_dir() else sorted(d for d in p.iterdir() if (d / "package").is_dir())
        if not candidates:
            raise SystemExit(f"no packaged task under {p}")
        out += candidates
    return out


def register(task_dirs: list[Path], dep_id: str) -> Deployment:
    specs, predictors, paths, exp_ids = {}, {}, {}, []
    for d in task_dirs:
        spec = TaskSpec.model_validate_json((d / "task.json").read_text())
        if spec.id in specs:
            raise SystemExit(f"two packages for task id '{spec.id}': pass only one of them")
        paths[spec.id] = str((d / "package").resolve())
        predictors[spec.id] = ml_api.load_predictor(paths[spec.id])
        specs[spec.id] = spec
        best = d / "best" / "result.json"
        exp_ids.append(best.exists() and __import__("json").loads(best.read_text())["exp_id"] or "?")
        print(f"  {spec.id:<30} {spec.prediction_format:<12} {paths[spec.id]}")
    dep = Deployment(id=dep_id, run_id=",".join(sorted({d.parent.name for d in task_dirs})), task_ids=list(specs),
                     exp_ids=exp_ids, package_paths=paths, endpoint=f"/api/deployments/{dep_id}")
    registry.deployments[dep_id] = registry.LiveDeployment(dep, specs, predictors)
    return dep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run dirs (all packaged tasks) or task dirs")
    p.add_argument("--id", default="dep_served")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"deployment {a.id}:")
    register(packages(a.paths), a.id)
    uvicorn.run(app, host="127.0.0.1", port=a.port)
