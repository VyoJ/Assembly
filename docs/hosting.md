# Hosting: backend on Modal (CPU), frontend on Vercel

```
browser ──> Vercel (static Vite build) ──fetch / SSE──> Modal: FastAPI in one container ──> Token Factory (LLM)
                                                              │  volume: data/, runs/          └─> Daytona (train.py)
```

## Backend (Modal)

One container serves the whole API (`modal_app.py`): runs, approvals and deployments live in that process, so
`max_containers=1`. It **scales to zero** when idle; the first request after a pause cold-starts a container
(usually under a minute; the frontend shows a "waking up" banner). Every run is recorded on the volume, so after a
cold start past runs replay and their deployments reload on first use.

```bash
modal token new                                      # once, if the CLI isn't logged in
uv run python scripts/modal_setup.py                 # secret from .env + data/ + runs/ (not runs/archive) to the volume
uv run modal deploy modal_app.py                     # prints the URL: https://<workspace>--assembly-api-web.modal.run
uv run python scripts/modal_setup.py --only runs     # after recording a new demo run locally
```

- **Secret** `assembly-secrets`: the settings from `.env` (LLM, Daytona key, budgets), never paths or mock flags.
  `EXECUTOR=daytona` by default (training in Daytona sandboxes keeps the Modal container small);
  `--executor local` trains inside the Modal container instead.
- **Volume** `assembly-data`: `/vol/data` (C-MAPSS, CWRU, MVTec, weights) and `/vol/runs`. YOLO training scratch
  (`exp_*/yolo`) isn't uploaded. Paths recorded on a laptop are rebased onto the volume (`ml.workspace.relocate`).
- **Image**: the same pinned versions as the local environment with CPU torch (pyproject's Linux torch is ROCm).
- **Limits to know**: web requests are capped at 150 s by Modal; event streams that get cut reconnect and the
  UI de-duplicates the replay. A pipeline run keeps going while the UI is open (its polling keeps the container
  busy) and for 20 minutes after the last request; closing every tab for longer can stop a run mid-way (it then
  shows as "interrupted").
- **Cost** (Starter plan, $30/month free credits): 2 vCPU + 4 GiB while a container is up, about $0.08/hour;
  nothing while scaled to zero (the volume is storage only).

## Frontend (Vercel)

`frontend/vercel.json` builds the Vite app and rewrites every route to `index.html`. In the Vercel project:

- **Root directory**: `frontend`
- **Environment variable**: `VITE_API_BASE_URL=https://<workspace>--assembly-api-web.modal.run` (no trailing slash)

```bash
cd frontend && npx vercel                  # first time: log in and link the project, then a preview deploy
npx vercel --prod
```

The browser calls the Modal URL directly (the API allows any origin), so event streams aren't proxied.
