# Assembly Mission Control: user and developer guide

Mission Control is the web frontend for Assembly. You give it an industrial goal and watch the agent work through it:
it formulates ML tasks, prepares data, waits for your approval, runs experiments, evaluates the winner and deploys
it. You never need the terminal.

This guide covers how to use the UI, how it talks to the backend, and what was built. For setup details and
scripts, see [README.md](README.md).

---

## 1. Quick start

```bash
# Terminal 1, repo root: the backend. Mock mode needs no GPU and no LLM server.
ML_MOCK=1 LLM_MOCK=1 uv run uvicorn api.main:app --port 8000

# Terminal 2: the frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The sidebar's **Backend** box should show **API online**. It also shows **ML mock** /
**LLM mock** badges when those modes are on. In mock mode every metric is simulated, and the UI says so.

To use a real model, start the backend with your usual `.env` (`ML_MOCK=0`, `LLM_*` set). The frontend needs no
changes.

---

## 2. Using the app

### Overview (`/`)

The landing page shows:

- **Counters**: active runs, runs awaiting approval, completed runs, deployments.
- **Approval banner**: appears when any run is waiting for you. **Review plan** opens it.
- **Active runs** and **Run history**: click a run to open its workspace.
- **Deployments**: models registered by finished runs.

> Runs and deployments live in the API process's memory. Restarting the backend clears them. Workspace files under
> `runs/` stay on disk.

### Start a run (`/runs/new`)

1. Describe the outcome you want in plain language, and put the KPIs in the sentence, for example:
   *"Predict press failures on Line 3 with at least 90% recall and fewer than 5% false alarms."*
   The example chips fill the box for you.
2. Choose how the plan gets approved:
   - **Manual approval** (default, recommended): the agent pauses after designing the data plan so you can review it
     before any training.
   - **Auto-approve**: research starts as soon as the plan is ready.
3. Click **Start run** (or press ⌘/Ctrl + Enter). This sends `POST /api/runs` and opens the new run's workspace.

### The run workspace (`/runs/:id`)

The header shows the goal, the run id, the **run status** (Running / Awaiting approval / Done / Failed / Rejected /
No tasks), the **stream status** (Live / Reconnecting / Stream closed) and the elapsed time.

Below it, the **stepper** tracks the eight pipeline stages:

| Stage | What the agent does |
|---|---|
| Formulate | Turns the goal into ML task specs |
| Discover | Profiles plant streams and keeps the relevant ones |
| Design | Writes `prepare.py` and validates the dataset it builds |
| Approve | Waits for you (or approves itself with auto-approve) |
| Research | Autoresearch loop: propose a change to `train.py`, run it, score it, keep or discard |
| Full train | Retrains the winner with a bigger budget |
| Final gate | Scores the winner on the hidden test split |
| Deploy | Packages the models and registers a deployment |

Each step shows its real duration, taken from event timestamps. If a run stops early, the step where it stopped is
marked **Stopped here** or **Failed here**, and later steps are marked **Not reached**.

The workspace has five tabs.

#### Activity

- One **card per task**: framing, prediction format, current phase, streams kept/scanned, `prepare.py` status and
  attempts, experiment counts, best experiment, and KPI targets (green/red once an experiment has been scored).
- While research runs, the card shows the experiment in flight and its latest progress values.
- Exploration notes and the research plan written by the agent can be expanded.
- On the right, the **Live activity** feed lists every agent step as it happens. Tick **tool calls & progress** to
  include the noisier events.

#### Plan & approval

When the run reaches **Approve** with manual approval, this tab opens automatically. For each task you see:

- The specification: rationale, framing, target definition, split strategy, input streams and label sources.
- The **stream profiles** (relevance and why each stream was kept or dropped) and every **dataset validation**
  attempt with its checks.
- The **data card** written during design.

You can change only what the approval API applies:

- **Primary metric**, **metrics to report**, **KPIs** (metric, ≥/≤, threshold; add or remove), and the
  **time budget per experiment**. The backend clamps the budget to its `MIN_BUDGET_S`/`MAX_BUDGET_S` bounds.
- **Drop task** to exclude a task from research.

Then click **Approve plan**, or **Reject** with an optional reason. After approval the tab shows the final, frozen
specs, so you can confirm your edits were applied. If someone else resolved the approval first, the UI tells you
instead of failing silently.

Before the approval step, the tab shows the draft specs. For auto-approved runs it says **Plan auto-approved**.

#### Experiments

Pick a task (if there are several) to see:

- **Summary**: total, kept, discarded and failed experiments, the best experiment, and the best value of the primary
  metric.
- **Experiment history chart**: any validation metric (or the combined score) per experiment. Point colour shows the
  outcome (green = keep/new best, grey = discard, red = crash/timeout/invalid). A dashed line marks the KPI
  threshold when one targets that metric.
- **Table**: score, primary metric, KPIs passed, the change made, train time. The ★ marks the best experiment.
- **Detail panel** (click a row): hypothesis, change, all metrics, KPI results, and, for experiments seen live, the
  `train.py` **diff** and the **log tail**.

> **Score is lower-is-better.** It combines a large penalty per failed KPI with the primary metric
> (`ml/scoring.py`). The best experiment is the last one marked `keep`.

#### Results

Shown when the run reaches the final stages:

- The **deployment** card, linking to the deployment page.
- The **agent report** written after the final gate.
- Per task: the **hidden test split** metrics and pass/fail with any failed KPIs, the full-train validation metrics,
  and why research stopped.

#### Event log

Every raw event, newest first. Filter by type or search the payloads, click a row to see its JSON, and use **JSONL**
to download the whole log.

### Deployments (`/deployments`)

Select a deployment to see its run, endpoint, tasks and package paths. Then:

- **Inference tester**: edit a JSON sample and click **Predict** (`POST /api/deployments/{id}/predict`). Templates
  match what the predictor accepts: replay a held-out test sample (`{"replay": 0}`), named feature values, an image
  path, or in mock mode `{"risk": 0.85}`. You can target all tasks or one. Each result shows the main output (score
  or value), **ALARM / no alarm**, latency and any other fields.
- **Live simulator**: **Start** streams samples from the simulated plant through every model in the deployment
  (`GET /api/deployments/{id}/live`, at 0.5–5 Hz). You get a chart of each model's output with alarm points and
  the running alarm rate. **This is simulated data from `plant/simulator.py`, not real plant monitoring.**

Deployments are registered automatically at the end of a run. The backend does not yet block deployment when the
final gate fails.

### Data sources (`/data`)

- Filter the **stream catalog** (grouped by asset, e.g. `press3`, `motor4`) and select a stream to see its source,
  unit and sample rate.
- The preview chart is **synthetic**. `GET /api/streams/{id}/preview` returns values generated by the plant
  simulator, not rows from the dataset. The API has no raw-data endpoint yet.
- **Label sources** lists the ground truth the agent can build targets from.
- The catalog is configured on the server (`plant/catalog.yaml`). You cannot upload data from the UI.

### When things go wrong

| Situation | What you see |
|---|---|
| Backend down | Yellow banner at the top, **API offline** in the sidebar, "Backend unavailable" panels with **Retry**. Recovers on its own when the API comes back. |
| Event stream drops | **Reconnecting** badge. Events already on screen are kept and the stream resumes without duplicates. |
| Backend restarted mid-run | **Run not found**: the run lived in memory and is gone. |
| Run failed | Red callout with the backend's error; the stepper marks where it failed. |

---

## 3. How it works

```
Browser ──► Vite dev server (:5173) ──proxy /api, /health──► FastAPI (:8000)
              │
              ├─ REST: runs, approval, experiments, streams, deployments, predict
              └─ SSE:  /api/runs/{id}/events (agent steps), /api/deployments/{id}/live (simulator)
```

- **Same origin.** The dev server proxies `/api` and `/health`, so there are no CORS issues and SSE streams pass
  through. If the backend is unreachable, the proxy returns 502 so the UI can say "backend unavailable".
  `ASSEMBLY_API_TARGET` changes the target; `VITE_API_BASE_URL` skips the proxy.
- **Events drive the workspace.** `useRunEvents` reads the run's SSE stream with `fetch`. The backend uses each
  event's type as the SSE event name, which `EventSource` can only handle with one listener per type. A pure reducer
  (`src/lib/runModel.ts`) folds the events into the per-task view model. Unknown event types still appear in the
  event log.
- **Reconnects without duplicates.** The backend replays the full history on every connection and events have no
  id. After a reconnect, the client skips the events it already has by position, checked by timestamp and type
  (`mergeReplayed` in `src/api/sse.ts`). The stream is closed after `run_finished` / `run_failed`.
- **Status from REST.** `awaiting_approval` is only exposed by `GET /api/runs/{id}`, so the workspace polls it every
  3 s while the run is live and refetches right after key events. Polling stops once the run is finished.
- **Experiments.** The table comes from `GET /api/runs/{id}/tasks/{task}/experiments` and is refetched whenever an
  `experiment` event arrives. Diffs and log tails come from the live events, because the endpoint omits them.
- **Only real data.** Nothing is invented: no progress percentages, no fabricated metrics, and no buttons for
  features the backend doesn't have.

---

## 4. What was built

| Area | Delivered |
|---|---|
| Foundation | Vite + React 19 + TypeScript + Tailwind v4, dark design tokens, responsive layout with sidebar (desktop) and menu (mobile), route-level code splitting |
| API layer | Typed client mirroring `core/contracts.py` and the API payloads, typed run events, fetch-based SSE client with backoff, health polling |
| Screens | Overview, New Run, Run Workspace (stepper and five tabs), Deployments (tester and simulator), Data Sources, 404 |
| Approval | Editing limited to the fields the approval API applies; metric choices limited to names the spec already uses so edits aren't silently ignored; drop, reject with reason, handling for a run that is no longer waiting |
| Charts | Experiment history with KPI threshold lines, simulator output per task, stream previews, all with tooltips |
| Reliability | Loading, empty and error states on every screen; offline banner; reconnect without duplicates; "run not found" after a backend restart |
| Tests | 13 Vitest unit tests: event reducer, stage states, best-experiment selection, SSE parser, replay merge |

**Verification**: `npm run typecheck`, `npm run lint` (no warnings), `npm test` (13/13) and `npm run build` all pass.
The full flow was exercised in a browser against the backend in mock mode:

- create a run from the UI;
- watch the live stream (UI and server event counts matched exactly);
- approve with an edited time budget (confirmed applied on the backend);
- reject with a reason, and an auto-approved run;
- experiments, results, inference and the simulator;
- data sources, mobile width (390 px), backend down and recovery.

Not yet tested against a real LLM or real training (`ML_MOCK=0`).

No backend code was changed.

---

## 5. Known limits (backend)

These are not faked in the UI. Each would need backend work:

- Runs and deployments are kept in memory only and are lost on restart.
- SSE events have no ids, so every reconnect replays the full history.
- The metric registry (and metric directions) and the budget bounds are not exposed by the API.
- There is no endpoint for previewing raw dataset rows; previews are synthetic.
- No cancel, pause or resume for runs; no authentication; no data upload.
- Deployment is not gated on the final test result.
- The experiments endpoint does not return diffs or log tails.

---

## 6. Code map

```
frontend/src/
├── api/          client.ts (REST), sse.ts (SSE client + replay merge), types.ts, events.ts
├── hooks/        useApi (fetch/poll), useRunEvents (run stream), health (backend status)
├── lib/          runModel.ts (events → view model) + tests, utils.ts
├── components/   Layout, status badges, chart helpers, ui/ primitives
├── features/run/ Stepper and the five workspace tabs
└── pages/        Overview, NewRun, Run, Deployments, DataSources, NotFound
```
