# Assembly: Implementation Plan

The core of Assembly is an **autoresearch-style loop** (after [karpathy/autoresearch](https://github.com/karpathy/autoresearch)). There are **no predefined task types or templates**. From the business goal, the agent decides:
- what ML problems to solve
- how to build the dataset and target from the plant's streams and logs
- which models and frameworks to use
- how much compute each experiment gets

It writes all the code itself. One thing is fixed: **how predictions are scored**. If the agent wrote its own evaluation, it would be grading its own homework.

Each task runs in two phases, mirroring autoresearch's fixed `prepare.py` + editable `train.py`:

1. **Design.** The agent writes a `TaskSpec` (target, prediction format, metrics, KPIs, budget) and a `prepare.py` that assembles the dataset and splits it. Deterministic validators check it for leakage and sanity, and the human approves. Then `prepare.py` is **frozen** and run once. Labels for the val/test splits go to a hidden location.
2. **Research.** The agent writes `train.py` from scratch (`exp_000`) and iterates. It edits, runs under a time budget, and gets scored by fixed scoring code against labels it can't see. A better score is kept and a worse one is discarded. The loop repeats until the KPIs are met or the budget runs out.

Hackathon rules of thumb:

1. **Build the end-to-end skeleton first, using mocks.** The whole pipeline (goal → design → research → deploy → live stream) should run on fake data before any real model exists.
2. **One contract between the two halves.** The agent only calls `ml.api` (`src/ml/api.py`), and both sides speak the pydantic types in `src/core/contracts.py`.
3. **The LLM is creative; scoring is strict.** The LLM writes `prepare.py` and `train.py` and decides what to try. Validation, execution, scoring and keep/discard are deterministic Python.
4. **Keep a demo fallback.** Record a real session so the demo can replay it if anything breaks live.

---

## 0. Current plan (updated 16 Oct)

Three changes from the original plan below:
- **LLM:** AMD **Token Factory** shared endpoints ([docs/Token-Factory.md](docs/Token-Factory.md)), not self-hosted vLLM.
- **Compute:** **Daytona** sandboxes ($300 credits), not AMD Developer Cloud instances (credits not guaranteed).
- **Order:** get one **real end-to-end run with the LLM agent** on the ML that exists today, then build out.

Base branch: `feature/ml-integration` (everything from `init`, `ml-data` and `ml-profiling-validation`).

### Key decisions

1. **The agent, API and scoring run outside Daytona; only agent-written code runs inside it.** Daytona Tier 1–2
   sandboxes only reach PyPI, GitHub and Hugging Face (full internet needs Tier 3, a $500 top-up), so a sandbox
   can't call Token Factory. This split also gives real sandboxing for generated code, and hidden labels never
   leave the host. Only `ml/runner.py` changes: all execution goes through `_run()`.
2. **Phase 1 runs on a laptop CPU.** C-MAPSS + tree models is small; Daytona adds risk, not capability, at this stage.
3. **The first real run is sensors only** (C-MAPSS RUL / failure-within-N). It's the only data kind with a working
   hand-written path and a recipe. Hide the rest with `CATALOG_EXCLUDE`.

### Token Factory facts that shape the code

- Base URL `https://developer.amd.com.cn/radeon/api/v1`, one `rc-…` key **shared by the whole team**.
- Limits per key: ~20 requests/min, 8 concurrent, and a **$1/day spend cap** (resets 00:00 Asia/Shanghai).
  Check with `scripts/check_llm.py` or `GET /v1/usage`. Don't run big batches while someone is recording the demo.
- Structured output: only MiMo accepts `json_schema`, so `LLM_STRUCTURED_METHOD=function_calling`.
- Default model `DeepSeek-V4-Flash` (no thinking by default, 1M context, tool calling). Fallbacks: `GLM-5.3-Flash`,
  `Qwen3.8-27B`. Thinking tokens count toward `max_tokens`; empty content means raise `LLM_MAX_TOKENS`.
- The catalog changes: `check_llm.py` verifies `LLM_MODEL` against `GET /v1/models`.
- Prompts and outputs are logged for ~2 days. Nothing sensitive.
- **Open question:** the Evolus track requires our own vLLM on AMD Developer Cloud; Token Factory may rule it out.

### Phase 1: real LLM, full run, local CPU (now)

Setup: `.env` per `.env.example` (`LLM_*`, `ML_MOCK=0`, `BUDGET_PROFILE=demo`), then
`uv run python scripts/download_data.py --only cmapss`.

Done in code: `agent/llm.py` retries 429/5xx with backoff, caps concurrent calls (`LLM_MAX_CONCURRENCY`, default 3),
sets `max_tokens`, timeout and optional `reasoning_effort`; `CATALOG_EXCLUDE` hides streams / label sources.

Run in steps, adding one LLM-written piece at a time:

| Step | Command | Proves |
|---|---|---|
| a | `uv run python scripts/check_llm.py` | Key, URL, structured output, tool calls |
| b | `run_research.py --spec specs/rul_example.json --prepare specs/rul_example_prepare.py --n 5` | The LLM writes a `train.py` that runs and beats its own baseline |
| c | Same without `--prepare` | The LLM writes a `prepare.py` that passes the validators |
| d | `CATALOG_EXCLUDE=… run_pipeline.py` with a sensor-only goal | formulate → discover → design → research → full train → final test → deploy |
| e | Same through the API with approval (`--no-auto-approve` + approve endpoint) | Approval pause + SSE events |

Expect to iterate `program.md` and the prompts at b and c; that's the real work of this phase. If deploy breaks
because the simulator sends synthetic values a real model can't use, demo with the predictor's `{"replay": i}` mode.
Keep `runs/{run}/events.jsonl` from the first good run as the first fallback demo.

### Phase 2: experiments on Daytona

1. `ml/executor.py`, `EXECUTOR=local|daytona`. Under Daytona, per task: create one sandbox from a prebuilt snapshot
   at freeze; upload `data/` (inputs only) + `harness.py`. Per experiment: upload `train.py`, run it as an async
   session command with the time limit, stream logs to progress events, download predictions / timings / `model/`,
   score locally. `prepare.py` stays local (cheap, and the only step that needs labels).
2. Snapshot: `uv sync --group ml` + C-MAPSS / CWRU data + weights baked in (the C-MAPSS S3 download isn't reachable
   from Tier 1–2 sandboxes).
3. CPU sandboxes first (default max 4 vCPU / 8 GiB / 10 GiB disk per sandbox; enough for sensors). Ephemeral or
   `auto_delete_interval=0`; raise `auto_stop_interval` above the 15-minute default during research.
4. GPU for vision: Daytona has **AMD Instinct MI355X** (fits the AMD story; needs a ROCm image, untested) and NVIDIA
   (H100 / RTX Pro 6000; CUDA image, lower risk). `gpu_type` takes an ordered preference list.
5. Check first: Daytona says free credits can't pay for GPU sandboxes; confirm what the $300 is, and GPU prices on
   the billing dashboard. On-demand GPUs queue for capacity; spot GPUs need Tier 2 and can be preempted.

### Phase 3: build out

1. Vision (ML team): `specs/metal_nut_example.json` + `_prepare.py`, image anomaly recipe, cached weights; then two
   tasks in parallel.
2. More recipes and `program.md` tuning, driven by what the real LLM gets wrong.
3. Frontend (not started): plan review, research feed, score chart, live dashboard.
4. Simulator replays real C-MAPSS / MVTec test data.
5. Record a strong `BUDGET_PROFILE=demo` run; rehearse.

**Cut:** SAM 3 + pseudo-labels, PatchTST, `Dockerfile.runtime`, self-hosted vLLM, stretch goals.

> Sections 1–9 below are the original plan. Where they mention vLLM or AMD Developer Cloud instances, section 0 wins.

---

## 1. Setup (do this together, ~1 hour)

```bash
# repo root (done): hatch backend so src/core, src/api, src/agent, src/ml, src/plant are top-level packages
uv init --package --name assembly --python 3.12 --build-backend hatch .

# core / API / agent
uv add fastapi "uvicorn[standard]" pydantic pydantic-settings sse-starlette httpx python-dotenv
uv add langgraph langchain-core langchain-openai

# ML: this is also the agent's toolbox, since generated code can only import installed packages
uv add numpy pandas pyarrow scikit-learn xgboost lightgbm optuna pillow opencv-python-headless \
       onnx onnxruntime joblib matplotlib tqdm torchmetrics pycocotools
uv add torch torchvision          # see the ROCm section below
uv add ultralytics                # YOLO (AGPL-3.0, fine for a hackathon)
uv add transformers accelerate huggingface_hub timm   # SAM 3, PatchTST, backbones

uv add --dev pytest ruff ipykernel
```

Adding a package later widens what the agent can do. That's the only change needed: `data_card.md` lists installed packages automatically.

### ROCm PyTorch via uv

With this in `pyproject.toml`, Linux (the AMD box) pulls ROCm wheels and macOS keeps the default PyPI wheels:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"   # match the box's ROCm version; check pytorch.org
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-rocm", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-rocm", marker = "sys_platform == 'linux'" }]
triton-rocm = [{ index = "pytorch-rocm", marker = "sys_platform == 'linux'" }]  # torch dep, only on the ROCm index
```

The ML stack lives in the `ml` dependency group (`uv sync --group ml` on the AMD box); `triton-rocm` is listed there
explicitly (Linux only) because uv applies `[tool.uv.sources]` only to packages you declare.

**Fallback:** use the `rocm/pytorch` Docker image + `uv venv --system-site-packages`.

### LLM serving

The agent **writes all the code**, so use the strongest coding-capable open model you can serve. Run it with **vLLM on the AMD GPU** (`vllm serve <model> --port 8001`, which is OpenAI-compatible) and call it through `ChatOpenAI(base_url=..., model=...)`. Leave GPU memory for training runs: with vLLM, `--gpu-memory-utilization 0.5` or similar. For laptop development, point the same env vars at a hosted OpenAI-compatible provider.

### Compute (AMD Developer Cloud)

Everything (vLLM, design and research runs, full training, deployment, and the FastAPI backend) runs on AMD cloud GPU instances. Laptops are only for editing code and mock-mode development.

**Credits:** $100 per person, about $200 total. Check the per-GPU-hour price on the console. At roughly $2/hour for a single MI300X, that's on the order of **~100 GPU-hours** total. That's plenty, but an instance left running all night eats about 15% of it.
- Plan on **one single-GPU instance per person**, each serving its own vLLM plus experiments. Get one box working end to end first, and keep the second for parallel development or as a demo box.
- **Destroy instances when idle.** On some clouds, a stopped instance still bills; check this on day one. Keep `data/`, model weights and `runs/` reproducible from `scripts/download_data.py` + git, or snapshot them.
- Do all agent/API/UI development with `ML_MOCK=1` on laptops (vLLM is the only remote dependency). Use GPU time for real ML work.

**Per-experiment time budget.** The agent chooses it in the `TaskSpec`, e.g. 3 min for a tree model on sensors or 10 min for a detector. It must fall within `[MIN_BUDGET_S, MAX_BUDGET_S]`, and the human sees it at approval. The budget is fixed for the whole research phase. That isn't about cost:
1. **Comparability:** every experiment gets the same compute, so "better score" means "better idea", not "trained longer".
2. **Loop speed:** the agent learns from each result.
3. **Demo:** the research feed has to move while judges watch.

The real compute goes to a **full-train stage**: the winning `train.py` is re-run with `FULL_TRAIN_MULTIPLIER`× the budget. `BUDGET_PROFILE=demo` caps all budgets low for live demos.

**Deployment** lives on the same AMD box. The API (and the predictor endpoints) are served from there and exposed either on the instance's public IP with a firewall rule, or via a tunnel (e.g. `cloudflared`). The frontend points at that URL.

### Housekeeping
- `.gitignore`: `.venv/`, `data/`, `runs/`, `.env`, `__pycache__/`, `*.ipynb_checkpoints`, `node_modules/`
- `.env.example` with the variables from the README

---

## 2. Architecture

```
                         frontend (React, later): goal form · plan review · research feed · live dashboard
                                                   │ REST + SSE
┌──────────────────────────────────────── backend ──────────────────────────────────────────────┐
│ api/  POST /runs ─► asyncio task runs the OUTER graph; every event → SSE + runs/{id}/events.jsonl │
│                                                                                               │
│ agent/  OUTER GRAPH (LangGraph)                                                               │
│   formulate → discover → design (per task) → approve_plan → research (per task)               │
│            → full_train → final_gate → deploy                                                 │
│                                                                                               │
│   DESIGN SUBGRAPH:   write prepare.py → run → validate (no LLM) → fix (≤3) → freeze           │
│   RESEARCH SUBGRAPH: propose (LLM + read-only tools) → apply edit → run → judge (no LLM)      │
│                      → crash? fix (≤2) → stop? (KPIs met · budget · plateau) → loop           │
│                                                                                               │
│ ml/  plant catalog · harness · validators · runner · scoring (metric registry) · recipes      │
│      program.md · building blocks · package (ONNX/joblib) · predictor                         │
│ plant/  simulator: C-MAPSS sensors + MVTec camera + maintenance/QC logs + distractor streams  │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
          PyTorch on ROCm (AMD Instinct GPU) runs experiments · vLLM on the same box runs the agent's LLM
```

### What's fixed vs. what the agent decides

| Fixed (our code) | Decided by the agent (its code) |
|---|---|
| Plant catalog: streams + label sources (maintenance log, QC inspections) | Which problems to solve, framing (regression vs. classification vs. detection…), target definition |
| `harness.py` API: how data is read and written, how predictions are submitted | `prepare.py`: joins, windowing, resampling, target construction, split strategy |
| **Prediction formats** + **metric registry** + scorer | Which metrics and KPIs (from the registry), primary metric, time budget |
| Validators (leakage, sizes, balance), runner, keep/discard rule | `train.py`: features, model, framework, training procedure, thresholds, ensembling |
| Installed packages, pre-downloaded weights | Which packages/weights to use |

### Mapping autoresearch → Assembly

| autoresearch | Assembly |
|---|---|
| `prepare.py` (fixed, human-written) | `prepare.py` **written by the agent in the design phase**, validated, approved by the human, then **frozen** before any scores exist |
| `train.py` (only editable file) | `train.py` per task; the agent also writes the first version (`exp_000`) |
| `program.md` (human-written) | One generic `program.md` written by the ML teammate, plus this task's `task.json` |
| `val_bpb`, lower is better | `score` from the fixed scorer: primary metric + penalty for each KPI violation |
| 5-minute budget | Budget per task, chosen by the agent within bounds and fixed during research |
| git commit / reset | snapshot per experiment in `exp_NNN/` + a `best/` pointer |
| `results.tsv` | `results.tsv` + `experiments.jsonl`, streamed live to the UI |
| "NEVER STOP" | stops when KPIs pass on validation, the budget is spent, or there's no improvement in N tries |
| (not in autoresearch) | **Full-train** of the winner, then a **final gate on a hidden test split** the loop never sees |

### File map

```
src/
├── core/
│   ├── config.py              # LLM_*, ML_MOCK, LLM_MOCK, DATA_DIR, RUNS_DIR, DEVICE, budget bounds, MAX_EXPERIMENTS
│   ├── contracts.py           # SHARED schemas (§3)
│   ├── events.py              # event bus → SSE + runs/{id}/events.jsonl
│   ├── approvals.py           # human approval futures
│   └── registry.py            # live deployments + predictors
├── api/
│   └── main.py · runs.py · streams.py · deployments.py
├── agent/
│   ├── state.py               # OuterState, DesignState, ResearchState
│   ├── graph.py               # outer graph + run_pipeline()
│   ├── llm.py · prompts.py · mock_llm.py   # LLM calls (each with a canned LLM_MOCK fallback)
│   ├── tools.py               # read-only tools: list_catalog, peek_stream, read_recipe, read_experiment_file
│   ├── nodes/                 # plan.py (formulate, discover) · build.py (design, approve_plan, research)
│   │                          # ship.py (full_train, final_gate, deploy)
│   ├── design/graph.py        # subgraph: write_prepare → run_prepare (validate) → retry ≤ MAX_DESIGN_ATTEMPTS
│   └── research/              # graph.py (propose → run → fix → check_stop), edits.py, judge.py
├── plant/
│   ├── simulator.py · catalog.yaml   # streams AND label sources
└── ml/
    ├── api.py                 # PUBLIC FACADE (dispatches to mock.py if ML_MOCK=1)
    ├── mock.py
    ├── catalog.py             # load plant data as stream tables / image folders + label-source tables
    ├── profiling.py           # stream stats, constant-channel detection, correlation vs. a candidate target
    ├── workspace.py           # create/freeze workspaces + experiment bookkeeping (keep/discard, best/, results.tsv)
    ├── validators.py          # checks on prepare.py output (§5)
    ├── runner.py              # run prepare.py / train.py in a subprocess with timeout, then score
    ├── scoring.py             # prediction formats + METRIC REGISTRY + scalar score
    ├── package.py             # best experiment → deployable package + Predictor
    ├── harness/harness.py     # copied read-only into every workspace
    ├── program.md             # generic research instructions (one file for every task)
    ├── recipes/               # reference code the agent can READ (not templates) + INDEX.md
    └── models/                # small set of project-specific building blocks the agent can import
scripts/
├── download_data.py           # datasets + ALL model weights (YOLO, SAM 3, timm) so experiments never download
├── run_pipeline.py            # headless outer graph (main dev loop for the agent teammate)
└── run_research.py            # headless design+research on one hand-written TaskSpec (main dev loop for the ML teammate)
```

### A task workspace

```
runs/{run_id}/{task_id}/
├── task.json         r/o   TaskSpec (frozen at approval)
├── program.md        r/o   generic research instructions + rules
├── prepare.py        r/o   written by the agent in design, frozen at approval, run once
├── data_card.md      r/o   generated after prepare.py runs: shapes, columns, label stats, split sizes,
│                           installed packages + versions, available weights
├── harness.py        r/o   (see below)
├── train.py          rw    THE ONLY FILE EDITABLE DURING RESEARCH
├── data/                   train inputs + labels; val/test INPUTS ONLY (written via the harness)
├── best/                   copy of the best experiment (train.py, metrics, model)
├── exp_000/ … exp_NNN/     train.py snapshot, run.log, metrics.json, model/, proposal.json
└── results.tsv             exp  score  primary  kpis_passed  latency_ms  status  description
# hidden, outside the workspace: runs/{run_id}/_labels/{task_id}/{val,test}.*
```

**`harness.py`:**
- **For `prepare.py`:** `catalog()` (streams + label sources), `read_stream(id)`, `read_labels(source)`, `write_dataset(train=…, val=…, test=…, groups=…, format=…)`. `write_dataset` puts val/test labels in the hidden location.
- **For `train.py`:** `load_train()`, `load_val_inputs()`, `load_test_inputs()`, `task()`, `time_left()`, `device()`, `weights(name)`, `save_model(model)`, `finish(predict_fn)`.

`finish(predict_fn)` runs the model on val/test inputs, checks that the output matches the task's **prediction format**, measures p50/p95 latency, and writes predictions. The **runner** computes metrics against the hidden labels. During research the agent only ever sees validation scores.

---

## 3. The contract (`contracts.py`), agreed on first

```python
from typing import Literal
from pydantic import BaseModel

Modality = Literal["video", "image", "vibration", "pressure", "temperature", "plc", "log", "other"]

# The ONLY fixed vocabulary: how predictions look, so the fixed scorer can grade them.
PredictionFormat = Literal[
    "scalar",        # one float per sample (regression, RUL, …)
    "binary_score",  # one score per sample + threshold (failure in N cycles, anomaly, defect present…)
    "class_probs",   # probability vector per sample (failure mode, …)
    "boxes",         # per image: [x1,y1,x2,y2,score,class] list
    "masks",         # per image: instance masks + scores
]

class StreamInfo(BaseModel):
    id: str; name: str; modality: Modality; unit: str | None = None
    source: str; sample_hz: float | None = None

class LabelSource(BaseModel):         # things a target can be built from
    id: str; name: str; description: str   # "maintenance_log: unit failures with timestamps", "qc_inspections: defect annotations"
    kind: Literal["events", "annotations", "table"]

class KPI(BaseModel):
    metric: str                       # must exist in the metric registry for this prediction format
    op: Literal["<=", ">="]; threshold: float

class TaskSpec(BaseModel):            # written by the agent, approved by the human, then frozen
    id: str; name: str
    rationale: str                    # why this serves the business goal
    framing: str                      # free text: "predict cycles to failure per engine from sensor history"
    input_streams: list[str]
    label_sources: list[str]
    target_definition: str            # free text, implemented by prepare.py
    prediction_format: PredictionFormat
    primary_metric: str               # from the registry
    metrics: list[str]                # from the registry
    kpis: list[KPI]
    split_strategy: str               # "by engine unit", "by time", "stratified by defect type" … (validated)
    time_budget_s: int                # per experiment, within config bounds

class ProblemSpec(BaseModel):
    goal: str; tasks: list[TaskSpec]

class StreamProfile(BaseModel):
    stream_id: str; relevance: float; keep: bool; reason: str

class ValidationReport(BaseModel):
    ok: bool; checks: dict[str, bool]; messages: list[str]; data_card: str

class Workspace(BaseModel):
    task_id: str; path: str; spec: TaskSpec; frozen: bool

class Proposal(BaseModel):            # LLM output in research/propose
    hypothesis: str; description: str
    edits: list[dict]                 # [{"old": "...", "new": "..."}], or [{"rewrite": "<full file>"}]

class ExperimentResult(BaseModel):
    exp_id: str; task_id: str
    status: Literal["keep", "discard", "crash", "timeout", "invalid_output"]
    score: float | None               # lower is better
    metrics: dict[str, float]         # validation metrics only
    kpis_passed: dict[str, bool]
    train_seconds: float; peak_mem_gb: float | None
    description: str; hypothesis: str
    log_tail: str; diff: str

class FinalReport(BaseModel):
    task_id: str; exp_id: str; test_metrics: dict[str, float]; passed: bool; failed_kpis: list[str]

class Deployment(BaseModel):
    id: str; task_ids: list[str]; exp_ids: list[str]; endpoint: str; status: Literal["live", "stopped"]
```

### `ml/api.py`: what the agent calls

```python
def list_catalog() -> tuple[list[StreamInfo], list[LabelSource]]: ...
def metric_registry() -> dict[PredictionFormat, list[str]]: ...         # shown to the LLM in formulate
def profile_streams(spec: TaskSpec, streams: list[StreamInfo]) -> list[StreamProfile]: ...
def create_workspace(run_id: str, spec: TaskSpec) -> Workspace: ...
def run_prepare(ws: Workspace, prepare_py: str) -> ValidationReport: ...   # runs + validates; repeatable until frozen
def freeze(ws: Workspace) -> Workspace: ...
def read_workspace_file(ws: Workspace, name: str) -> str: ...
def run_experiment(ws: Workspace, train_py: str, meta: dict, progress=None) -> ExperimentResult: ...
#   writes exp_NNN/, subprocess with spec.time_budget_s (hard kill at 2×), checks format, scores vs hidden val
#   labels, keep (updates best/ + train.py) or discard. Deterministic.
def full_train(ws: Workspace, multiplier: float = 5.0, progress=None) -> ExperimentResult: ...
def final_test(ws: Workspace) -> FinalReport: ...
def package(ws: Workspace) -> str: ...
def load_predictor(package_path: str) -> "Predictor": ...             # .predict(sample) -> format-shaped output + latency
```

**Day-one rule:** `mock.py` implements everything with instant fake numbers. Its `run_prepare` sometimes fails validation, and its `run_experiment` returns noisily improving scores and crashes about 10% of the time, which exercises every fix path. The agent teammate builds the whole flow against it, with real LLM calls.

---

## 4. Agent side

### Outer graph
```
START → formulate      LLM: goal + catalog (streams, label sources) + metric registry + package list
                       → ProblemSpec: 1..N TaskSpecs (structured output; metrics/KPIs must exist in the registry)
      → discover       ml.profile_streams per task + LLM reason per kept/dropped stream → updates input_streams
      → design         design subgraph per task (concurrent) → validated prepare.py + data_card.md
      → approve_plan   interrupt(): the human sees each TaskSpec + data card and approves or edits → freeze
      → research       research subgraph per task (concurrent)
      → full_train     ml.full_train per task
      → final_gate     ml.final_test per task; LLM writes report vs. the original business goal
      → deploy         ml.package + register Deployment + start live monitor → END
```

### Design subgraph
- **write_prepare** (LLM + read-only tools: `list_catalog`, `peek_data(stream, n)`, `read_recipe`). Writes `prepare.py` that uses `harness.read_*` and ends with `harness.write_dataset(...)`.
- **run + validate** → `ml.run_prepare` (deterministic validators, §5).
- **fix** (LLM gets the validation messages, ≤3 tries). If it still fails, revise the TaskSpec once, then drop the task and say why.

### Research subgraph (the autoresearch part)

`ResearchState`: `workspace, best, history, n_exp, started_at, fix_attempts, stop_reason`.

- **propose.** An LLM with a **small set of read-only tools** (`read_recipe`, `read_file` for past experiments' code and logs, `peek_data`). It gets at most ~5 tool calls, then must return a `Proposal`. Its context is rebuilt compactly each iteration, with no growing chat history:
  - `program.md`
  - `task.json`
  - `data_card.md`
  - `recipes/INDEX.md`
  - the current best `train.py`
  - the last ~20 rows of `results.tsv`
  - the last result's metrics, KPI status and log tail

  **For `exp_000`, it writes `train.py` from scratch**, usually a simple, fast baseline (`program.md` says to start simple). After that: one idea per experiment, with a hypothesis.
- **apply edits** (`edits.py`): search/replace on the best `train.py`, with full rewrite allowed. Only `train.py` is ever written.
- **run** → `await asyncio.to_thread(ml.run_experiment, ...)`, streaming progress as custom events.
- **judge** (no LLM): keep if the score beats the best, else discard. `crash`/`timeout`/`invalid_output` → **fix** (≤2 tries, then logged and skipped).
- **stop?** Stop on any of these:
  - all KPIs pass on val (optionally `EXTRA_AFTER_PASS` more)
  - `MAX_EXPERIMENTS`
  - `RESEARCH_BUDGET_S`
  - `PLATEAU_N` experiments in a row without improvement

The agent has full freedom over code. The loop's control flow, the context size and the evaluation stay fixed.

### Streaming
`graph.astream(..., stream_mode=["updates", "custom"], subgraphs=True)` → `api/events.py` → SSE. Key events:
- `{"type": "task_spec", ...}`
- `{"type": "validation", ...}`
- `{"type": "experiment", "task_id", "exp_id", "status", "score", "metrics", "hypothesis", "diff"}`

### Safety (hackathon level)
Agent-written code runs as a subprocess with `cwd` = the workspace, a timeout, and an env without API keys. The hidden-labels path is never passed to the research subprocess (it is passed to `prepare.py`, which is frozen before any scoring). Note it as a known limitation in the pitch. Stretch goal: run experiments in a throwaway container.

---

## 5. ML side

The ML teammate builds **the lab, not the experiments**: data catalog, harness, validators, scorer, recipes, building blocks, and the generic `program.md`. Then they make the agent good at using them.

### Plant catalog (demo data)
| Catalog entry | Backed by | Notes |
|---|---|---|
| ~21 sensor streams on "Press Line 3" (vibration, pressure, temperature…) + operating settings | **NASA C-MAPSS FD001** (NASA Prognostics Data Repository / Kaggle mirror) | ~6 sensors are constant, which gives discovery an easy visible win |
| `maintenance_log` label source (unit failure events) | Derived from C-MAPSS run-to-failure ends | The agent decides whether that becomes RUL regression, "fails within N cycles", or something else |
| `line3.cam_top` image stream | **MVTec AD**, one category with plenty of defects (e.g. `metal_nut`) | Manual download (license) |
| `qc_inspections` label source (defect masks/boxes per image) | MVTec ground-truth masks (+ boxes derived from them) | The agent decides between detection, segmentation, image-level classification, or anomaly scoring. It may also ignore these labels and use SAM 3 |
| Distractor streams (HVAC ambient temp, shift counter, noise) | Synthetic | Discovery should reject them |

Alternative image dataset with native boxes: NEU-DET (steel surface defects).

### Models we expect the agent to reach for
These are **not hard-coded**. They're what's installed, what `recipes/` shows, and what `program.md` hints at:
- **YOLO** (Ultralytics): fast detector/segmenter that meets latency KPIs.
- **SAM 3** (Meta): text-prompted open-vocabulary segmentation. The agent can use it as a zero-shot detector, or as an auto-labeler that produces YOLO training data ("SAM 3 labels, YOLO ships").
- **XGBoost / LightGBM / sklearn**: strong, fast sensor baselines.
- **Torch NNs** (LSTM/GRU, TCN, PatchTST via `transformers`): what the agent tries when the trees plateau.
- Anything else installed (PatchCore-style kNN on `timm` features, autoencoders, …).

### Deliverables, in priority order
1. **Plant catalog** (`catalog.py`, `plant/catalog.yaml`): C-MAPSS → sensor stream tables + `maintenance_log`; MVTec → image stream + `qc_inspections`; distractors.
2. **`harness.py`**: both the `prepare.py` side and the `train.py` side (§2).
3. **`scoring.py`: prediction formats + metric registry.** This is the fixed evaluation core, so make it broad, because KPIs can only use metrics listed here:

   | Format | Metrics |
   |---|---|
   | `scalar` | rmse, mae, r2, nasa_score, early_warning_recall@N, early_warning_far@N |
   | `binary_score` | auroc, pr_auc, recall, precision, f1, false_alarm_rate (at the submitted threshold) |
   | `class_probs` | accuracy, macro_f1, per-class recall |
   | `boxes` | map50, map50_95 (pycocotools/torchmetrics), image_recall, image_far |
   | `masks` | mask_map, iou, image_recall, image_far |
   | all formats | p50_latency_ms, p95_latency_ms |

   Score: `score = normalized_primary + 10 × Σ normalized KPI violations`, so meeting the KPIs always dominates.
4. **`validators.py`** (run on `prepare.py` output; this is what makes agent-designed tasks trustworthy):
   - **Splits:** no group overlap across splits when `groups` are given (e.g. engine unit, image id); time-ordered splits for time-based framing.
   - **Size and balance:** minimum sizes per split; label distribution sanity (no single-class val set, enough positives).
   - **Leakage and format:** no label column among the inputs, and inputs/labels shapes that match the `prediction_format`.
   - **Output:** generates `data_card.md`.
5. **`runner.py`**: subprocess, timeout, format check, parse metrics, peak memory (`torch.cuda.max_memory_allocated` works on ROCm), snapshot, keep/discard, `results.tsv`.
6. **`program.md`** (one generic file), the most important prompt in the project, iterated the way Karpathy iterates his. It covers:
   - how the score works, and to start with a simple fast baseline and change one idea per experiment
   - **rules:**
     - only edit `train.py`
     - **only import installed packages**: the list and versions are in `data_card.md`, so code targets the API that's actually installed
     - weights only via `harness.weights(...)`; no network
     - must call `harness.save_model` + `harness.finish`
     - size training with `time_left()`, so full-train just trains longer
     - don't touch the data files
   - a simplicity preference, and a pointer to `recipes/INDEX.md`
7. **`recipes/`**: short, working, **reference** scripts plus a one-line `INDEX.md` entry each. The agent can read them; it doesn't have to use them. They're where model-exploration knowledge goes:
   - `xgb_window_features.py`, `lstm_sequence_regressor.py`, `patchtst_hf.py`
   - `yolo_finetune.py`, `sam3_zero_shot.py`, `sam3_pseudolabel_then_yolo.py`, `patchcore_knn.py`
   - `threshold_calibration.py`, `optuna_inside_train.py`
   - `INDEX.md` also lists ideas: asymmetric RUL loss, piecewise RUL target, `imgsz` 1024 + tiling for small defects, SAM 3 prompt wording, etc.
8. **`ml/models/` building blocks** (below).
9. **`package.py`** + `Predictor` + `Dockerfile.runtime` (onnxruntime + FastAPI) for the "lightweight runtime container".

### Building blocks: use the packages directly

`train.py` and `prepare.py` are ordinary Python, so the agent uses **installed packages directly**: `xgboost`, `lightgbm`, `sklearn`, `torch.nn`, `ultralytics`, `transformers`, `pandas`. No wrappers for any of those. Write a block **only** if it is project-specific, easy to get silently wrong, slow/cacheable, error-prone boilerplate, or environment-specific:

| Block (`ml/models/`) | API sketch | Why it's a block |
|---|---|---|
| `features.py` | `window_features(df, group_col, window, stats=…, fft_bands=None)` | Windowing **per group** to avoid leakage |
| `calibrate.py` | `calibrate_threshold(scores, labels, max_far=…, min_recall=…)` | Many KPIs are won or lost here, and it's easy to get subtly wrong |
| `splits.py` | `internal_split(train, groups, frac=0.2)` | `train.py` has no val labels, so it needs a leakage-free internal split |
| `selfcheck.py` | Registry metrics on the internal split | Lets the agent sanity-check before `finish()` |
| `pseudolabel.py` | `sam3_to_yolo(image_dir, prompts, score_thr, out_dir)` → `data.yaml`, **cached by (prompts, thr)** | SAM 3 is slow, so cache it; project-specific format conversion |
| `sam3.py` | `load_sam3()`, `sam3_detect(images, prompts, score_thr)` | New model the LLM may not know well; local weights |
| `vision_utils.py` | `masks_to_boxes`, `tile_images`/`untile_preds`, `image_level_score` | Glue |
| `export.py` | `save_model(model)` → ONNX / joblib / SAM 3 config; `load_saved(path)` | Every kept experiment must be deployable; ONNX shapes are fiddly |
| `device.py` | `device()` → `cuda` (ROCm) / `cpu` | Environment-specific |

**Pre-download every weight file** (YOLO checkpoints, SAM 3, timm backbones) in `scripts/download_data.py` into `data/weights/`, and expose them via `harness.weights(name)`. `data_card.md` lists what's available.

**GPU-specific library checks (M0):**
- **XGBoost:** `device="cuda"` on ROCm depends on the build, and pip wheels may be CPU-only.
- **LightGBM:** needs a GPU build.
- **SAM 3:** check that it runs on ROCm at all.

If the trees don't run on GPU, `program.md` says they run on CPU, which is fast enough here.

`scripts/run_research.py --spec specs/rul_example.json --n 10` runs design + research on a hand-written TaskSpec. This is the ML teammate's main loop for tuning `program.md`, recipes, validators and the harness, without the API or outer graph.

### Optuna?
The agent can write an Optuna sweep *inside* `train.py` if it judges that worthwhile (there's a recipe). There's no separate HPO stage, because the research loop *is* the refinement step.

---

## 6. Work split

| | **Teammate A: Agent / Platform** | **Teammate B: ML / Research lab** |
|---|---|---|
| Owns | `api/`, `agent/` (outer graph, design + research subgraphs, tools, prompts except `program.md`), `plant/simulator.py`, `scripts/run_pipeline.py`, later `frontend/` | `ml/` (catalog, harness, validators, runner, scoring/metric registry, `program.md`, recipes, building blocks, packaging), `notebooks/`, `scripts/run_research.py`, `scripts/download_data.py` |
| First task | `ml/mock.py` + outer graph + both subgraphs running against mocks | C-MAPSS catalog + harness + runner + scorer for `scalar`/`binary_score` + a hand-written `prepare.py`/`train.py` pair that passes end to end |
| Interface | `ml/api.py` + `contracts.py` | same |

**Shared boundary:** Teammate A owns the *format* of prompts (Proposal schema, context assembly, tools). Teammate B owns the *research content* (`program.md`, recipes, metric registry docs).

---

## 7. Milestones

**M0: Setup (both, first ~1–2h)**
- [ ] uv project, deps, `.gitignore`, `.env.example`, `config.py`
- [ ] `contracts.py` + `ml/api.py` signatures agreed
- [ ] vLLM serving a coding-capable model on the AMD box (or hosted fallback) · ROCm torch verified
- [ ] **Request SAM 3 weight access on Hugging Face now** (gated, I believe)
- [ ] Smoke tests on the AMD box: YOLO train 1 epoch · SAM 3 on 1 image · XGBoost `device="cuda"`

**M1: Walking skeleton**
- A: [ ] `mock.py` · [ ] outer graph · [ ] design subgraph · [ ] research subgraph with **real LLM calls** against mock runs · [ ] `/runs` + SSE · [ ] simulator
- B: [ ] sensor catalog + `maintenance_log` · [ ] harness (both sides) · [ ] validators v1 (group overlap, sizes, format) · [ ] runner + scorer (`scalar`, `binary_score`) · [ ] first `program.md` · [ ] `features.py`, `splits.py`
- ✅ Exit: `ML_MOCK=1 run_pipeline.py` runs goal → specs → design → research → deploy; `run_research.py` runs a real hand-written sensor spec

**M2: Agent-designed sensor task, for real**
- B: [ ] recipes for sensors · [ ] `calibrate.py`, `selfcheck.py` · [ ] `export.py` (joblib + torch ONNX) · [ ] iterate `program.md` until the agent reliably writes a working `exp_000` and improves on it within ≤10 experiments
- A: [ ] wire real `ml.api` · [ ] read-only tools · [ ] edit-apply robustness · [ ] approve-plan interrupt showing spec + data card · [ ] concurrency across tasks
- ✅ Exit: from the goal text alone, the agent frames a sensor task, writes a valid `prepare.py`, and beats its own baseline to hit a KPI

**M3: Vision + deployment**
- B: [ ] image stream + `qc_inspections` · [ ] `boxes`/`masks` scoring · [ ] `sam3.py` + cached `pseudolabel.py` · [ ] vision recipes · [ ] YOLO ONNX export · [ ] `final_test` · [ ] `package` + `Predictor` + `Dockerfile.runtime`
- A: [ ] full-train + final gate + report nodes · [ ] `/deployments/{id}/live` streaming predictions + rolling KPI tracking
- ✅ Exit: both tasks designed and researched in parallel on the GPU, deployed, streaming live

**M4: Frontend + demo polish**
- A: [ ] React app with these views:
  - Goal form
  - **Plan review**: task specs, data cards, approve button
  - **Research feed**: hypothesis, diff, score, and keep/discard per experiment
  - Score-over-time chart
  - Live dashboard
- B: [ ] record a strong real session for replay fallback · [ ] SAM 3 vs. YOLO comparison numbers · [ ] GPU-vs-CPU throughput numbers for the pitch
- Both: [ ] pick demo goal + KPIs so the baseline fails and the agent passes after a few visible experiments · [ ] rehearse

---

## 8. Demo script (~3 min)

1. *"Prevent unplanned stoppages on Press Line 3. Max 1 false alarm per shift; catch ≥90% of failures."*
2. The agent frames two tasks on its own and shows them for approval:
   - "Will this press fail within 30 cycles?" from sensors + the maintenance log
   - "Is this part defective?" as detection from the camera + QC inspections

   Each comes with a target definition, split strategy, metrics, KPIs and budget, plus the data card from its own `prepare.py` → Approve.
3. Discovery: 30 streams scanned; it keeps 14 and drops HVAC and noise, with reasons.
4. **Research feed comes alive.** Each step shows a code diff, and both tasks run in parallel on one AMD GPU.
   - **Sensors:**
     - The agent's own baseline is XGBoost on window features, and it misses the false-alarm KPI.
     - "LSTM on 40-cycle windows" → keep
     - "threshold calibration" → **KPIs met**
   - **Camera:**
     - SAM 3 zero-shot is too slow.
     - YOLO-n on SAM 3 pseudo-labels misses small scratches.
     - `imgsz` 1024 + tiling → **KPIs met**
5. Full-train the winners → final gate on the hidden test split → deploy → live dashboard with streaming alerts, camera frames with detected defects, and a KPI tracker.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent frames a task badly (wrong target, leaky split) | Validators + human approval of spec and data card before freezing; recipes show good `prepare.py` patterns |
| Agent writes broken code | Start-simple rule, recipes, building blocks, small-edit preference, crash-fix paths |
| Agent games the metric | Fixed scorer + metric registry, labels hidden outside the workspace, `prepare.py` frozen before any scores, hidden test split for the final gate |
| More flexibility makes the live demo less predictable | Pin a demo goal that you've rehearsed; recorded-session replay fallback |
| Code written against the wrong library version | Installed packages + versions in `data_card.md`; pinned `uv.lock` |
| SAM 3 doesn't run on ROCm, or weight access isn't granted in time | Check in M0; fallback is YOLO on QC labels only |
| Too few defect images | Augmentation, tiling, SAM 3 pseudo-labels; or NEU-DET |
| Experiments too slow for a live demo | `BUDGET_PROFILE=demo`, small `MAX_EXPERIMENTS`, replay |
| Burning through cloud credits | One box per person, destroy when idle, mock mode for agent/UI dev |
| vLLM + training compete for GPU memory | Cap vLLM memory utilization |
| Running untrusted generated code | Subprocess + timeout + no secrets in env; container sandbox as a stretch goal |
| Ultralytics AGPL-3.0 | Fine for a hackathon; mention if asked about commercial use |

## 10. Stretch goals
Parallel proposals per round (population search across GPU memory) · container sandbox per experiment · the agent proposes **new metrics** as code that a human reviews before they enter the registry · drift detection that triggers a new research session · RTSP via mediamtx · SAM 3 video tracking on camera streams · a "steer the agent" chat box that appends to `program.md` mid-run.
