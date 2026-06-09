"""
Frontend configuration — V10 schema (2-class engaged / not-engaged).
"""

import os

# FastAPI backend URL — set via env var or default to local dev
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
PAGE_TITLE = "Classroom Engagement Analyzer"
PAGE_ICON = "🎓"
LAYOUT = "wide"

# Upload limits (must match backend)
MAX_VIDEO_SIZE_MB = 500
ALLOWED_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# Polling interval for processing status (seconds)
STATUS_POLL_INTERVAL = 3

# ── Engagement display config (2-class) ──────────────────────────────────

ENGAGEMENT_COLORS = {
    "engaged":     "#2ecc71",
    "not-engaged": "#e74c3c",
}

ENGAGEMENT_LABELS = {
    "engaged":     "Engaged",
    "not-engaged": "Not Engaged",
}

ENGAGEMENT_EMOJI = {
    "engaged":     "🟢",
    "not-engaged": "🔴",
}

# Engagement colors for charts (dark mode aware)
ENGAGEMENT_CHART_COLORS = {
    "engaged":     "#34d399",
    "not-engaged": "#fb7185",
}
