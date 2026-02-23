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
from components.styles import inject_global_css, hero_section, feature_card, init_theme

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

init_session_state()
init_theme()
inject_global_css()

# ── Sidebar ───────────────────────────────────────────────────────────────
show_user_sidebar()

# ── Main view ─────────────────────────────────────────────────────────────
if is_logged_in():
    hero_section(
        title=PAGE_TITLE,
        subtitle="Analyze student engagement from classroom videos using AI-powered detection, tracking, and classification.",
        emoji=PAGE_ICON,
    )

    # ── Feature cards ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        feature_card(
            emoji="📤",
            title="Upload Video",
            description="Upload a classroom video and let AI analyze each student's engagement level.",
            accent="#6366f1",
        )

    with col2:
        feature_card(
            emoji="📊",
            title="View Results",
            description="See detailed per-student engagement reports with interactive charts and annotated video.",
            accent="#34d399",
        )

    with col3:
        feature_card(
            emoji="📋",
            title="History",
            description="Browse all your past analyses, compare results, and download reports.",
            accent="#fbbf24",
        )

    st.markdown("")

    # ── Quick health check ────────────────────────────────────────────
    from services.api_client import APIClient

    try:
        health = APIClient().health()
        if health.get("models_loaded"):
            st.success(f"✅ Backend online — models loaded (device: {health.get('device', 'N/A')})")
        else:
            st.warning("⏳ Backend is starting up — models are still loading…")
    except Exception:
        st.error("❌ Cannot reach the backend API. Make sure FastAPI is running on port 8000.")
else:
    show_auth_page()
