# Assembly

**An autonomous AI engineer for intelligent factories.**

Give Assembly a business goal ("prevent line stoppage on Line 3") and a KPI ("≤5% false alarms, catch ≥90% of failures"). The agent decides on its own what ML problems to solve and how to frame them, finds the plant data streams that matter, and writes the data-prep code. There are no predefined task types or model templates. It then runs an **autoresearch-style loop** (after [karpathy/autoresearch](https://github.com/karpathy/autoresearch)) on AMD ROCm GPUs: it rewrites the training code, runs a time-boxed experiment, is scored by evaluation code it can't touch, and keeps or discards the change. Once the KPIs are met it deploys a live prediction endpoint. You don't write code or need ML expertise.

```
 Goal + KPI ─► 1. Formulate tasks ─► 2. Discover data ─► 3. Design: agent writes prepare.py → validate → you approve → freeze
                                                          │
                     ┌────────────────────────────────────┘
                     ▼
        4. Research loop:  agent writes/edits train.py ─► run (time budget) ─► score vs hidden labels
                     ▲                                                     │
                     └──────────── keep if better, else discard ◄──────────┘   until KPIs met / budget spent
                                                          │
                                                          ▼
                5. Full-train winner ─► final gate on hidden test split ─► package + deploy ─► live monitoring
```

## Repo layout

```
Assembly/
├── README.md · PLAN.md
├── pyproject.toml · uv.lock · .python-version · .env.example
├── src/
│   ├── core/               ← config.py (env vars), contracts.py (SHARED agent <-> ML schemas), event bus,
│   │                         approvals, deployment registry
│   ├── api/                ← FastAPI app + routes                            (owner: Agent teammate)
│   ├── agent/              ← LangGraph: outer graph, design + research subgraphs, prompts, tools
│   │                                                                          (owner: Agent teammate)
│   ├── plant/              ← catalog.yaml (streams + label sources) + live simulator
│   └── ml/                 ← the "research lab": catalog, harness, workspace bookkeeping, runner, scoring /
│       │                     metric registry, validators, program.md, recipes, building blocks, packaging
│       │                                                                      (owner: ML teammate)
│       ├── api.py          ← the ONLY module the agent imports from ml/
│       └── mock.py         ← fake ML layer used when ML_MOCK=1
├── scripts/                ← run_pipeline.py, run_research.py, download_data.py
├── specs/                  ← hand-written TaskSpecs for run_research.py
├── tests/
├── notebooks/              ← model exploration
├── frontend/               ← React web app (later)
├── data/                   ← (gitignored) datasets + model weights
└── runs/                   ← (gitignored) per-run task workspaces: train.py, exp_NNN/, results.tsv, events.jsonl
```

`src/core`, `src/api`, `src/agent`, `src/ml` and `src/plant` are top-level packages (installed in editable
mode by `uv sync`), so imports look like `from ml import api as ml_api` and `from core.contracts import TaskSpec`.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + Uvicorn, SSE for live agent and prediction streams |
| Orchestration | LangGraph: outer pipeline graph + an autoresearch-style inner loop (LLM proposes code edits; runner and judge are deterministic) |
| LLM | Any OpenAI-compatible endpoint. Target: a strong open coding model served by vLLM on an AMD Instinct GPU |
| ML | YOLO (Ultralytics), SAM 3 (Meta), XGBoost/LightGBM/scikit-learn, PyTorch (ROCm) NNs, Optuna (available to the agent), ONNX Runtime for deployment |
| Data (demo) | NASA C-MAPSS turbofan (sensor RUL), MVTec AD with masks→boxes (defect detection), replayed as "plant streams" |
| Frontend | Vite + React + TypeScript (later) |

## Quickstart

```bash
uv sync                                   # core + dev deps (enough for mock mode on a laptop)
cp .env.example .env                      # ML_MOCK=1 by default; set LLM_* to use a real model

# Whole pipeline headless. LLM_MOCK=1 also fakes the LLM (no model server needed).
ML_MOCK=1 LLM_MOCK=1 uv run python scripts/run_pipeline.py

# Design + research loop on one hand-written spec
ML_MOCK=1 LLM_MOCK=1 uv run python scripts/run_research.py --spec specs/rul_example.json --n 10

# API → http://localhost:8000/docs
uv run uvicorn api.main:app --reload --port 8000

uv run pytest                             # mock-mode end-to-end + unit tests
```

With `ML_MOCK=0`, anything not built yet fails with `NotImplementedError("TODO(ML): ...")` naming the function to implement.

### On the AMD GPU box (ROCm)

Training, research runs, vLLM, and deployment all run on AMD Developer Cloud GPU instances. Laptops run mock mode only. **Destroy instances when idle** to save credits.

The heavy ML stack (torch, ultralytics, transformers, xgboost, …) is the `ml` dependency group. On Linux, torch, torchvision and triton-rocm come from the PyTorch ROCm index (`pyproject.toml`: `[tool.uv.sources]`; match the index to the box's ROCm version):

```bash
uv sync --group ml
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ROCm builds of PyTorch use the `torch.cuda` API, so `device="cuda"` works unchanged. On a Mac, the same code runs on `mps` or `cpu`.

## Environment variables

| Var | Purpose | Example |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint | `http://gpu-box:8001/v1` (vLLM) |
| `LLM_MODEL` | Model name served there | any instruct model with tool calling |
| `LLM_API_KEY` | API key (vLLM accepts any string) | `sk-local` |
| `ML_MOCK` | `1` = ML layer returns fake instant results | `0` |
| `LLM_MOCK` | `1` = canned LLM responses (no model server) | `0` |
| `LLM_STRUCTURED_METHOD` | `json_schema`, `function_calling` or `json_mode`, whatever your server supports | `json_schema` |
| `BUDGET_PROFILE` | `research` (the agent picks each task's per-experiment budget within bounds) or `demo` (caps it low) | `research` |
| `MIN_BUDGET_S` / `MAX_BUDGET_S` | Bounds for the agent's per-experiment budget | `60` / `900` |
| `FULL_TRAIN_MULTIPLIER` | Extra budget for retraining the winning config | `5` |
| `MAX_EXPERIMENTS` | Max experiments per task per run | `20` |
| `RESEARCH_BUDGET_S` | Total research wall clock per task | `7200` |
| `DATA_DIR` / `RUNS_DIR` | Where datasets and artifacts live | `./data`, `./runs` |
| `DEVICE` | `cuda` (ROCm), `mps`, `cpu`, or `auto` | `auto` |

## API (summary)

| Method | Path | What |
|---|---|---|
| `POST` | `/api/runs` | Start an agent run from `{goal, auto_approve?}` → `{run_id, status}` |
| `GET` | `/api/runs/{id}` | Status (`running` / `awaiting_approval` / `done` / …) + full state |
| `GET` | `/api/runs/{id}/events` | **SSE** stream of agent steps, plus per-experiment hypothesis, diff, score, and keep/discard |
| `GET` | `/api/runs/{id}/tasks/{task_id}/experiments` | The research log (results.tsv as JSON) |
| `POST` | `/api/runs/{id}/approve` | `{approved, tasks?, drop?}`: approve/edit/drop the agent's task specs |
| `GET` | `/api/streams` | Plant streams + label sources |
| `GET` | `/api/streams/{id}/preview` | Sample values of a stream |
| `GET` | `/api/deployments` | Deployed models |
| `POST` | `/api/deployments/{id}/predict` | One-off prediction |
| `GET` | `/api/deployments/{id}/live` | **SSE** live predictions on replayed plant data, plus KPI tracking |

## Team

- **Agent / platform** owns `api/`, `agent/`, `plant/`, and later `frontend/`.
- **ML** owns `ml/` (the lab the agent researches in: plant catalog, harness, validators, metric registry, recipes, `program.md`), `notebooks/`, and the data and research scripts.
- **Shared**: `src/core/contracts.py`. Change it only after a quick sync between the two of you.

See **[PLAN.md](PLAN.md)** for the detailed build plan.
