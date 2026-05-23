"""
Upload page — select a video file and submit for processing.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import requests as _requests

import time
import streamlit as st

from components.auth import require_auth, get_api_client, show_user_sidebar
from components.styles import inject_global_css, hero_section, section_header, card, init_theme
from fe_config import (
    PAGE_TITLE, PAGE_ICON, ALLOWED_EXTENSIONS,
    MAX_VIDEO_SIZE_MB, STATUS_POLL_INTERVAL,
)
from i18n import t

st.set_page_config(page_title=f"Upload | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

hero_section(
    title=t("upload_title"),
    subtitle=t("upload_subtitle", MAX_VIDEO_SIZE_MB),
    emoji="📤",
)

# ── Instructions ──────────────────────────────────────────────────────────

section_header(t("how_it_works"), "⚡")

col1, col2, col3 = st.columns(3)
with col1:
    card(f'<div style="text-align:center;">'
         f'<div style="font-size:1.8rem;margin-bottom:0.4rem;">1️⃣</div>'
         f'<div style="font-weight:600;margin-bottom:0.2rem;">{t("step1_title")}</div>'
         f'<div style="font-size:0.85rem;opacity:0.7;">{t("step1_desc")}</div>'
         f'</div>')
with col2:
    card(f'<div style="text-align:center;">'
         f'<div style="font-size:1.8rem;margin-bottom:0.4rem;">2️⃣</div>'
         f'<div style="font-weight:600;margin-bottom:0.2rem;">{t("step2_title")}</div>'
         f'<div style="font-size:0.85rem;opacity:0.7;">{t("step2_desc")}</div>'
         f'</div>')
with col3:
    card(f'<div style="text-align:center;">'
         f'<div style="font-size:1.8rem;margin-bottom:0.4rem;">3️⃣</div>'
         f'<div style="font-weight:600;margin-bottom:0.2rem;">{t("step3_title")}</div>'
         f'<div style="font-size:0.85rem;opacity:0.7;">{t("step3_desc")}</div>'
         f'</div>')

# ── File uploader ─────────────────────────────────────────────────────────

section_header(t("select_video"), "🎬")

uploaded = st.file_uploader(
    "Choose a video file",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded is not None:
    st.video(uploaded)
    st.caption(f"📁 {uploaded.name} — {uploaded.size / 1024 / 1024:.1f} MB")

btn_disabled = bool(st.session_state.get("tracking_analysis_id")) or st.session_state.get("uploading", False)
if uploaded is not None and st.button(
    t("btn_analyze"), type="primary", use_container_width=True, disabled=btn_disabled
):
    st.session_state["uploading"] = True
    api = get_api_client()

    with st.spinner("Mengunggah video ke server…"):
        try:
            resp = api.upload_video(uploaded, uploaded.name)

        except _requests.exceptions.ReadTimeout:
            st.session_state["uploading"] = False
            st.warning(t("upload_timeout"))
            if st.button(t("btn_open_history"), type="primary"):
                st.switch_page("pages/3_History.py")
            st.stop()

        except _requests.exceptions.ConnectionError:
            st.session_state["uploading"] = False
            st.error(t("upload_conn_err"))
            st.stop()

        except Exception as e:
            st.session_state["uploading"] = False
            st.error(t("upload_fail", e))
            if st.button(t("btn_check_history"), key="err_history"):
                st.switch_page("pages/3_History.py")
            st.stop()

    st.session_state["uploading"] = False
    analysis_id = resp["analysis_id"]
    st.session_state["last_analysis_id"] = analysis_id
    st.session_state["tracking_analysis_id"] = analysis_id

# ── Processing tracker (shown after upload, stays on this page) ───────────

tracking_id = st.session_state.get("tracking_analysis_id")
if tracking_id:
    st.divider()
    section_header(t("processing_status_title"), "⏳")

    api = get_api_client()
    try:
        status_data = api.get_status(tracking_id)
        job_status = status_data.get("status", "processing")
    except Exception:
        job_status = "processing"

    if job_status == "completed":
        st.success(t("processing_done"))
        col_res, col_hist = st.columns(2)
        with col_res:
            if st.button(t("btn_view_results"), type="primary", use_container_width=True, key="tracker_results"):
                st.switch_page("pages/2_Results.py")
        with col_hist:
            if st.button(t("btn_go_history"), use_container_width=True, key="tracker_history"):
                st.session_state.pop("tracking_analysis_id", None)
                st.switch_page("pages/3_History.py")
        if st.button(t("btn_upload_another"), use_container_width=True, key="tracker_new"):
            st.session_state.pop("tracking_analysis_id", None)
            st.rerun()

    elif job_status == "failed":
        err = status_data.get("error_message", "Unknown error")
        st.error(f"{t('processing_failed')} {err[:200]}")
        if st.button(t("btn_upload_another"), use_container_width=True, key="tracker_retry"):
            st.session_state.pop("tracking_analysis_id", None)
            st.rerun()

    else:
        # Still processing — show simulated progress bar + auto-refresh
        from datetime import datetime, timezone
        try:
            created_raw = status_data.get("created_at", "")
            created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            elapsed_sec = int((datetime.now(timezone.utc) - created_dt).total_seconds())
        except Exception:
            elapsed_sec = 0

        if elapsed_sec < 60:
            elapsed_str = f"{elapsed_sec}s"
        else:
            elapsed_str = f"{elapsed_sec // 60}m {elapsed_sec % 60}s"

        # Simulated progress: caps at 95% until actually done
        # Assumes ~3 min average; slows down near the cap so it never freezes
        ASSUMED_DURATION = 180
        raw_pct = elapsed_sec / ASSUMED_DURATION
        progress_val = min(0.95, raw_pct * (1 - raw_pct * 0.3))
        progress_val = max(0.03, progress_val)

        st.progress(progress_val, text=t("processing_in_progress", elapsed_str))
        col_manual, col_hist = st.columns(2)
        with col_manual:
            if st.button(t("btn_refresh"), use_container_width=True, key="tracker_refresh"):
                st.rerun()
        with col_hist:
            if st.button(t("btn_go_history"), use_container_width=True, key="tracker_hist2"):
                st.session_state.pop("tracking_analysis_id", None)
                st.switch_page("pages/3_History.py")
        time.sleep(10)
        st.rerun()
