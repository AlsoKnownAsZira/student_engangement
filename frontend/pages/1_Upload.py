"""
Upload page — select a video file and submit for processing.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import time
import streamlit as st

from components.auth import require_auth, get_api_client, show_user_sidebar
from fe_config import (
    PAGE_TITLE,
    PAGE_ICON,
    ALLOWED_EXTENSIONS,
    MAX_VIDEO_SIZE_MB,
    STATUS_POLL_INTERVAL,
)

st.set_page_config(page_title=f"Upload | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
show_user_sidebar()

st.title("Upload Classroom Video")
st.markdown(
    f"Upload a classroom video (max **{MAX_VIDEO_SIZE_MB} MB**, formats: "
    f"{', '.join(ALLOWED_EXTENSIONS)}) and we'll analyze student engagement."
)

# ── File uploader ─────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Choose a video file",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
)

if uploaded is not None:
    st.video(uploaded)
    st.caption(f"{uploaded.name} — {uploaded.size / 1024 / 1024:.1f} MB")

    if st.button("Analyze Engagement", type="primary", use_container_width=True):
        api = get_api_client()

        # Upload
        with st.spinner("Uploading video…"):
            try:
                resp = api.upload_video(uploaded, uploaded.name)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        analysis_id = resp["analysis_id"]
        st.info(f"Processing started (ID: `{analysis_id}`)")

        # ── Poll for completion ───────────────────────────────────────────
        progress_bar = st.progress(0, text="Processing video…")
        status_placeholder = st.empty()

        tick = 0
        while True:
            time.sleep(STATUS_POLL_INTERVAL)
            tick += 1
            try:
                status_resp = api.get_status(analysis_id)
            except Exception:
                status_placeholder.warning("Lost connection — retrying…")
                continue

            current_status = status_resp["status"]

            if current_status == "completed":
                progress_bar.progress(100, text="Done!")
                st.success("Analysis complete!")
                # Store analysis_id for results page
                st.session_state["last_analysis_id"] = analysis_id
                st.switch_page("pages/2_Results.py")
                break

            elif current_status == "failed":
                progress_bar.empty()
                st.error(
                    f"Processing failed: {status_resp.get('error_message', 'Unknown error')}"
                )
                break

            else:
                # Indeterminate progress
                pct = min(95, tick * 5)
                progress_bar.progress(pct, text=f"Processing… ({current_status})")
