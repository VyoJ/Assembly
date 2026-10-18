# Assembly — frontend (mission control)

Vite + React 19 + TypeScript + Tailwind CSS v4, with Recharts for charts and lucide-react for icons. UI primitives
are small hand-written components in the shadcn style (`src/components/ui`), with no component library.

## Run it

```bash
# 1. Backend (repo root). Mock mode needs no GPU or LLM:
ML_MOCK=1 LLM_MOCK=1 uv run uvicorn api.main:app --port 8000

# 2. Frontend
cd frontend
npm install
npm run dev            # http://localhost:5173
```

The dev server proxies `/api` and `/health` to the backend, so the browser stays same-origin and SSE streams pass
straight through. Settings (`frontend/.env.local`, see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `ASSEMBLY_API_TARGET` | `http://localhost:8000` | Where the dev/preview proxy sends `/api` and `/health` |
| `VITE_API_BASE_URL` | *(empty)* | Call the backend directly instead of through the proxy (the backend allows any origin) |

## Scripts

| | |
|---|---|
| `npm run dev` | Dev server with proxy |
| `npm run build` | Typecheck + production build to `dist/` |
| `npm run preview` | Serve `dist/` (also proxies `/api`) |
| `npm run typecheck` / `npm run lint` / `npm test` | tsc, ESLint, Vitest (pure logic: event reducer, SSE parser, replay merge) |

## Screens

| Route | What | Backend |
|---|---|---|
| `/` | Overview: active runs, history, deployments, approvals waiting | `GET /api/runs`, `GET /api/deployments` |
| `/runs/new` | Goal in natural language + manual / auto approval | `POST /api/runs` |
| `/runs/:id` | Run workspace: stage stepper, per-task activity, live feed, plan and approval, experiments, results, raw event log | `GET /api/runs/{id}`, SSE `GET /api/runs/{id}/events`, `POST /api/runs/{id}/approve`, `GET /api/runs/{id}/tasks/{task}/experiments` |
| `/deployments/:id?` | Deployment details, inference tester, live **simulator** | `GET /api/deployments`, `POST /api/deployments/{id}/predict`, SSE `GET /api/deployments/{id}/live` |
| `/data` | Stream catalog, label sources, **synthetic** stream previews | `GET /api/streams`, `GET /api/streams/{id}/preview` |

## How the run workspace works

- **Events are the source of truth.** `src/hooks/useRunEvents.ts` reads the SSE stream with `fetch` (not
  `EventSource`), because the backend names each SSE event after its type and `EventSource` would need a listener
  per type. `src/lib/runModel.ts` folds events into a view model, a pure function covered by tests.
- **Reconnects.** The backend replays the full history on every connection and events carry no id, so after a
  reconnect the already-received prefix is skipped by position (`mergeReplayed` in `src/api/sse.ts`). The stream is
  closed after `run_finished` / `run_failed`. A 404 means the run is gone (the backend keeps runs in memory).
- **Status** comes from `GET /api/runs/{id}` (polled every 3 s while the run is live and refetched on key events),
  because `awaiting_approval` is only exposed there.
- **Approval** edits are limited to what the backend applies: `kpis`, `time_budget_s`, `primary_metric` and
  `metrics` per task, plus `drop` and a rejection `reason`. Metric choices are limited to names the task's spec
  already uses (the backend silently ignores edits with metrics outside its registry). The backend clamps the time
  budget to `MIN_BUDGET_S`/`MAX_BUDGET_S`.
- **Scores** are lower-is-better (`core/contracts.py`). The best experiment is the last one marked `keep`.

## Known backend limits surfaced in the UI

Runs and deployments survive API restarts as **recorded** runs: the API reads `runs/*/events.jsonl` back (replay only, marked "recorded"; a run that was killed shows as failed / interrupted), and a recorded run's deployment is listed as stopped and loads its packages on first use. There is no auth, cancel/pause, raw-data preview or upload endpoint.
The stream preview and the live endpoint use `plant/simulator.py` (synthetic data). Deployment is not gated on the
final test result.
