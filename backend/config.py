"""
Backend configuration — loads from environment variables / .env file.
"""

import os
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Supabase ---
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""              # anon/public key (for client-side auth)
    SUPABASE_SERVICE_KEY: str = ""      # service-role key (for server-side ops)

    # --- Model paths (relative to project root) ---
    DETECTION_MODEL_PATH: str = "models/yolo11s.pt"
    CLASSIFIER_MODEL_PATH: str = "models/best.pt"

    # --- Processing ---
    DEVICE: str = "auto"                # "auto", "0" (GPU), or "cpu"
    CONF_THRESHOLD: float = 0.3
    IOU_THRESHOLD: float = 0.5
    TRACKER_CONFIG: str = "botsort.yaml"
    MAX_VIDEO_SIZE_MB: int = 200
    ALLOWED_EXTENSIONS: str = ".mp4,.avi,.mov,.mkv"

    # --- Paths ---
    TEMP_DIR: str = "temp"
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)

    # --- Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # --- JWT (Supabase Auth) ---
    JWT_SECRET: str = ""
    JWT_ALGORITHM: str = "HS256"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # -- Derived helpers (not from env) --

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def max_video_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    @property
    def resolved_device(self) -> str:
        if self.DEVICE == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return self.DEVICE

    @property
    def temp_path(self) -> Path:
        p = Path(self.PROJECT_ROOT) / self.TEMP_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def detection_model_abs(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self.DETECTION_MODEL_PATH)

    @property
    def classifier_model_abs(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self.CLASSIFIER_MODEL_PATH)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    return Settings()
