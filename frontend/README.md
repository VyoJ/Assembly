# Frontend (later)

Vite + React + TypeScript. Talks to the backend at `http://<host>:8000/api`:

- `POST /api/runs` → start a run, then subscribe to `GET /api/runs/{id}/events` (SSE)
- `POST /api/runs/{id}/approve` → approve the agent's task specs
- `GET /api/deployments/{id}/live` (SSE) → live predictions

Generate TS types from the backend: `npx openapi-typescript http://localhost:8000/openapi.json -o src/api.d.ts`
