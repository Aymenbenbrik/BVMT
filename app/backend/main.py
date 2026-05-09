"""
FastAPI application — BVMT AI Platform backend.

Run with:
    uvicorn app.backend.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env (so EXPLAINER_LLM_* take effect)
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routers import stocks, prediction, graph, explain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic.  Agent models load lazily on first call."""
    print("[BVMT Backend] Starting up …")
    yield
    print("[BVMT Backend] Shutting down …")


app = FastAPI(
    title="BVMT AI Platform",
    description="Multi-agent stock analysis system for the Tunisian stock exchange",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — frontend runs on a different port ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ────────────────────────────────────────────────────
app.include_router(stocks.router)
app.include_router(prediction.router)
app.include_router(graph.router)
app.include_router(explain.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "bvmt-backend"}
