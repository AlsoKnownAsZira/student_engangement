"""
FastAPI entry point.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.services.pipeline_service import pipeline_manager
from backend.routers import auth, videos, results

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend")


# ── Lifespan — load models on startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models once when the server starts."""
    logger.info("Starting up — loading ML models …")
    pipeline_manager.load_models()
    logger.info("Startup complete ✓")
    yield
    logger.info("Shutting down …")


# ── App creation ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Engagement Analysis API",
    description="Upload classroom videos → get per-student engagement reports.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(results.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": pipeline_manager.is_ready(),
        "device": settings.resolved_device,
    }
