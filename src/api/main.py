"""uv run uvicorn api.main:app --reload --port 8000  ->  http://localhost:8000/docs"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import deployments, runs, streams
from core.config import settings

app = FastAPI(title="Assembly", description="Autonomous AI engineer for intelligent factories")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (runs.router, streams.router, deployments.router):
    app.include_router(r, prefix="/api")


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "ml_mock": settings.ml_mock, "llm_mock": settings.llm_mock, "llm_model": settings.llm_model}
