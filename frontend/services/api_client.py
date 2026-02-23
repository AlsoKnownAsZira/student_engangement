"""
HTTP client for communicating with the FastAPI backend.
"""

from __future__ import annotations
import sys
from pathlib import Path
import requests
from typing import Any, BinaryIO, Optional

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

from fe_config import API_BASE_URL


class APIClient:
    """Thin wrapper around ``requests`` that adds the auth header."""

    TIMEOUT = 15  # seconds — prevents Streamlit from hanging forever

    def __init__(self, token: str | None = None):
        self.base = API_BASE_URL.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # ── Auth ──────────────────────────────────────────────────────────────

    def signup(self, email: str, password: str, full_name: str = "") -> dict:
        r = requests.post(
            f"{self.base}/api/auth/signup",
            json={"email": email, "password": password, "full_name": full_name},
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def login(self, email: str, password: str) -> dict:
        r = requests.post(
            f"{self.base}/api/auth/login",
            json={"email": email, "password": password},
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def refresh(self, refresh_token: str) -> dict:
        r = requests.post(
            f"{self.base}/api/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    # ── Video upload ──────────────────────────────────────────────────────

    def upload_video(self, file: BinaryIO, filename: str) -> dict:
        r = requests.post(
            f"{self.base}/api/videos/upload",
            headers={"Authorization": f"Bearer {self.token}"},
            files={"file": (filename, file, "video/mp4")},
            timeout=120,  # uploads can be large
        )
        r.raise_for_status()
        return r.json()

    def get_status(self, analysis_id: str) -> dict:
        r = requests.get(
            f"{self.base}/api/videos/{analysis_id}/status",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    # ── Results ───────────────────────────────────────────────────────────

    def get_result(self, analysis_id: str) -> dict:
        r = requests.get(
            f"{self.base}/api/results/{analysis_id}",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def get_csv_url(self, analysis_id: str) -> str:
        r = requests.get(
            f"{self.base}/api/results/{analysis_id}/csv",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["csv_download_url"]

    def get_video_url(self, analysis_id: str) -> str:
        r = requests.get(
            f"{self.base}/api/results/{analysis_id}/video",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["output_video_url"]

    def get_history(self) -> dict:
        r = requests.get(
            f"{self.base}/api/results/",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def delete_analysis(self, analysis_id: str) -> None:
        r = requests.delete(
            f"{self.base}/api/results/{analysis_id}",
            headers=self._headers(),
            timeout=self.TIMEOUT,
        )
        r.raise_for_status()

    # ── Health ────────────────────────────────────────────────────────────

    def health(self) -> dict:
        r = requests.get(f"{self.base}/health", timeout=5)
        r.raise_for_status()
        return r.json()
