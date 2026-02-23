"""
Frontend configuration.
"""

import os

# FastAPI backend URL — set via env var or default to local dev
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
PAGE_TITLE = "Classroom Engagement Analyzer"
PAGE_ICON = "🎓"
LAYOUT = "wide"

# Upload limits (must match backend)
MAX_VIDEO_SIZE_MB = 200
ALLOWED_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# Polling interval for processing status (seconds)
STATUS_POLL_INTERVAL = 3

# Engagement level display config
ENGAGEMENT_COLORS = {
    "engaged": "#2ecc71",
    "moderately-engaged": "#f39c12",
    "disengaged": "#e74c3c",
}

ENGAGEMENT_LABELS = {
    "engaged": "Engaged",
    "moderately-engaged": "Moderately Engaged",
    "disengaged": "Disengaged",
}

ENGAGEMENT_EMOJI = {
    "engaged": "🟢",
    "moderately-engaged": "🟡",
    "disengaged": "🔴",
}
