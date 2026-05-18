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
from i18n import t

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
        subtitle=t("home_subtitle"),
        emoji=PAGE_ICON,
    )

    # ── Quick Stats ───────────────────────────────────────────────────
    from services.api_client import APIClient
    client = APIClient(st.session_state.get("access_token"))
    try:
        history_data = client.get_history()
        analyses = history_data.get("analyses", [])
        if analyses:
            total_videos = len(analyses)
            total_students = sum(a.get("total_students") or 0 for a in analyses)
            
            valid_scores = [a.get("avg_engagement_score") for a in analyses if a.get("avg_engagement_score") is not None]
            avg_engagement = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            st.markdown(f"### 📈 {t('quick_stats_title')}")
            st.markdown('<div style="margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(t("stat_total_videos"), total_videos)
            with m2:
                st.metric(t("stat_total_students"), f"{total_students} 👥")
            with m3:
                # Kalikan 100 karena data dari backend berupa pecahan (contoh: 0.8)
                st.metric(t("stat_avg_engagement"), f"{(avg_engagement * 100):.1f}%")
                
            st.markdown("<hr style='margin: 2rem 0; border-color: rgba(128,128,128,0.2);'/>", unsafe_allow_html=True)
    except Exception:
        pass

    # ── Feature cards ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        feature_card(
            emoji="📤",
            title=t("feat_upload_title"),
            description=t("feat_upload_desc"),
            accent="#38bdf8",
        )

    with col2:
        feature_card(
            emoji="📊",
            title=t("feat_results_title"),
            description=t("feat_results_desc"),
            accent="#34d399",
        )

    with col3:
        feature_card(
            emoji="📋",
            title=t("feat_history_title"),
            description=t("feat_history_desc"),
            accent="#a78bfa",
        )

    st.markdown("<hr style='margin: 2rem 0; border-color: rgba(128,128,128,0.2);'/>", unsafe_allow_html=True)

    # ── Quick Start Guide ─────────────────────────────────────────────
    st.markdown(f"### {t('quick_start_title')}")
    st.info(f"{t('qs_step1')}\n\n{t('qs_step2')}\n\n{t('qs_step3')}")
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Quick health check ────────────────────────────────────────────
    # APIClient is already imported above

    try:
        health = APIClient().health()
        if health.get("models_loaded"):
            raw_device = health.get("device", "N/A")
            device_str = str(raw_device).strip().lower()
            if device_str in ("cpu", "n/a", ""):
                device_label = "CPU" if device_str == "cpu" else "N/A"
            else:
                device_label = f"GPU (cuda:{device_str.replace('cuda:', '')})"
            st.success(t("backend_ok", device_label))
        else:
            st.warning(t("backend_loading"))
    except Exception:
        st.error(t("backend_error"))
else:
    show_auth_page()
