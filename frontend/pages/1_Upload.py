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

    if st.button(t("btn_analyze"), type="primary", use_container_width=True):
        api = get_api_client()

        with st.spinner("Mengunggah video ke server…"):
            try:
                resp = api.upload_video(uploaded, uploaded.name)

            except _requests.exceptions.ReadTimeout:
                st.warning(t("upload_timeout"))
                if st.button(t("btn_open_history"), type="primary"):
                    st.switch_page("pages/3_History.py")
                st.stop()

            except _requests.exceptions.ConnectionError:
                st.error(t("upload_conn_err"))
                st.stop()

            except Exception as e:
                st.error(t("upload_fail", e))
                if st.button(t("btn_check_history"), key="err_history"):
                    st.switch_page("pages/3_History.py")
                st.stop()

        analysis_id = resp["analysis_id"]
        st.success(t("upload_success", analysis_id))
        st.info(t("upload_redirecting"))
        time.sleep(2)
        st.session_state["last_analysis_id"] = analysis_id
        st.switch_page("pages/3_History.py")
