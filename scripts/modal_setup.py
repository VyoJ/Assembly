"""Prepare Modal for modal_app.py: the secret (from .env) and the volume (data/ + the demo runs).

    uv run python scripts/modal_setup.py                    # secret + data/ + every run in runs/ (not runs/archive)
    uv run python scripts/modal_setup.py --only secret
    uv run python scripts/modal_setup.py --only runs        # after adding a new demo run locally
    uv run python scripts/modal_setup.py --executor local   # train inside the Modal container instead of Daytona

The Modal container is the trusted side, like your laptop: runs/<id>/_labels (hidden labels) are uploaded because
the API scores against them. Agent-written code still never sees them (prepare.py runs next to them, train.py
doesn't get them; with EXECUTOR=daytona train.py runs in a Daytona box that never receives them).
The secret holds settings from .env only, never paths or mock flags (modal_app.py sets those).
Skipped: YOLO training scratch (exp_*/yolo, full/yolo: ~130 MB each, not needed to replay, score or deploy),
caches, the MVTec tar, runs/archive.
"""

import argparse
import sys
from pathlib import Path

import modal
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # modal_app.py lives at the repo root
from modal_app import SECRET_NAME, VOLUME_NAME

NOT_IN_SECRET = {"DATA_DIR", "RUNS_DIR", "ML_MOCK", "LLM_MOCK", "ISOLATE_PREDICTORS", "CATALOG_EXCLUDE"}
SKIP_DIRS = {"yolo", "__pycache__", "_scratch", "archive"}
SKIP_FILES = {"metal_nut.tar.xz", ".DS_Store"}


def secret(executor: str) -> None:
    env = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None and k not in NOT_IN_SECRET}
    if executor == "daytona" and not env.get("DAYTONA_API_KEY"):
        raise SystemExit("--executor daytona needs DAYTONA_API_KEY in .env")
    env["EXECUTOR"] = executor
    modal.Secret.objects.delete(SECRET_NAME, allow_missing=True)
    modal.Secret.objects.create(SECRET_NAME, env)
    print(f"secret {SECRET_NAME}: {sorted(env)}")  # names only, never values


def _files(base: Path):
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS) and p.name not in SKIP_FILES:
            yield p, rel


def upload(volume: modal.Volume, local: Path, remote: str) -> None:
    files = list(_files(local))
    size = sum(p.stat().st_size for p, _ in files) / 1e6
    print(f"upload {local.relative_to(ROOT)} -> {remote}: {len(files)} files, {size:.0f} MB")
    with volume.batch_upload(force=True) as batch:
        for p, rel in files:
            batch.put_file(str(p), f"{remote}/{rel.as_posix()}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--only", choices=["secret", "data", "runs"])
    a.add_argument("--executor", choices=["daytona", "local"], default="daytona")
    args = a.parse_args()
    if args.only in (None, "secret"):
        secret(args.executor)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    if args.only in (None, "data"):
        upload(vol, ROOT / "data", "/data")
    if args.only in (None, "runs"):
        for run in sorted(d for d in (ROOT / "runs").iterdir() if d.is_dir() and d.name not in SKIP_DIRS):
            upload(vol, run, f"/runs/{run.name}")
