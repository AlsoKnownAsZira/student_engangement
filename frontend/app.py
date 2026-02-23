"""
Classroom Engagement Analyzer — main Streamlit entry point.
"""

import sys
from pathlib import Path

# Ensure the frontend package root is importable
_FRONTEND_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
for _p in (_FRONTEND_DIR, _PROJECT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from fe_config import PAGE_TITLE, PAGE_ICON, LAYOUT
from components.auth import (
    init_session_state,
    is_logged_in,
    show_auth_page,
    show_user_sidebar,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────
show_user_sidebar()

# ── Main view ─────────────────────────────────────────────────────────────
if is_logged_in():
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown(
        """
        Welcome! Use the sidebar to navigate:

        - **Upload** — Upload a classroom video for engagement analysis
        - **Results** — View the latest analysis results
        - **History** — Browse all your past analyses
        """
    )

    # Quick health check
    from services.api_client import APIClient

    try:
        health = APIClient().health()
        if health.get("models_loaded"):
            st.success(f"Backend online — models loaded (device: {health.get('device', 'N/A')})")
        else:
            st.warning("Backend is starting up — models are still loading…")
    except Exception:
        st.error("Cannot reach the backend API. Make sure FastAPI is running.")
else:
    show_auth_page()
