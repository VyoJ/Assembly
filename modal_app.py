"""Assembly's backend on Modal (CPU only): the FastAPI app in ONE container that cold-starts on demand.

    uv run python scripts/modal_setup.py      # once (and after new demo runs): secret + data/ + runs/ on the volume
    uv run modal deploy modal_app.py          # -> https://<workspace>--assembly-api-web.modal.run
    uv run modal serve modal_app.py           # dev: hot-reloading temporary URL

Why one container: runs and deployments live in the API process (agent state, approvals, event bus), so every
request must reach the same process. Runs and their deployments are also recorded under RUNS_DIR on the volume,
so after a cold start the previous runs replay and deployments reload on first use.
A pipeline run is a background task inside this container: it keeps running while requests (the UI's polling
and event stream) keep the container busy, and for SCALEDOWN_S after the last request.

The image pins the same package versions as the local environment (so the pre-run API check matches) and
uses CPU torch: pyproject.toml's Linux source is ROCm torch, which is multi-GB and useless here.
"""

from pathlib import Path

import modal

from ml.executors.remote import ML_PACKAGES, TORCH_PACKAGES, pinned

APP_NAME = "assembly-api"
VOLUME_NAME = "assembly-data"
SECRET_NAME = "assembly-secrets"
VOL = "/vol"
SCALEDOWN_S = 20 * 60  # keep a container this long after the last request (a run must outlive short idle gaps)

API_PACKAGES = ["daytona", "fastapi", "httpx", "langchain-core", "langchain-openai", "langgraph", "sse-starlette",
                "uvicorn"]
APT_PACKAGES = ["libgl1", "libglib2.0-0", "libxcb1"]  # OpenCV (pulled in by ultralytics) needs these on Linux

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*APT_PACKAGES)
    .uv_pip_install(*pinned(TORCH_PACKAGES), index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install(*pinned(ML_PACKAGES), *pinned(API_PACKAGES))
    .env({
        "PYTHONPATH": "/root/src",
        "DATA_DIR": f"{VOL}/data",
        "RUNS_DIR": f"{VOL}/runs",
        "ML_MOCK": "0",
        "LLM_MOCK": "0",
        "ISOLATE_PREDICTORS": "off",  # Linux: no OpenMP clash, keep models in-process
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
    .add_local_dir(Path(__file__).parent / "src", "/root/src", ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    cpu=2.0,
    memory=4096,
    timeout=60 * 60,  # per request (the longest are event streams)
    max_containers=1,  # all state in one process: see the module docstring
    scaledown_window=SCALEDOWN_S,
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    from api.main import app as fastapi_app

    return fastapi_app
