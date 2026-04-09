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

st.set_page_config(page_title=f"Upload | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

hero_section(
    title="Upload Classroom Video",
    subtitle=f"Upload a video (max {MAX_VIDEO_SIZE_MB} MB) and let AI analyze student engagement",
    emoji="📤",
)

# ── Instructions ──────────────────────────────────────────────────────────

section_header("How It Works", "⚡")

col1, col2, col3 = st.columns(3)
with col1:
    card('<div style="text-align:center;">'
         '<div style="font-size:1.8rem;margin-bottom:0.4rem;">1️⃣</div>'
         '<div style="font-weight:600;margin-bottom:0.2rem;">Upload</div>'
         '<div style="font-size:0.85rem;opacity:0.7;">Select a classroom video file</div>'
         '</div>')
with col2:
    card('<div style="text-align:center;">'
         '<div style="font-size:1.8rem;margin-bottom:0.4rem;">2️⃣</div>'
         '<div style="font-weight:600;margin-bottom:0.2rem;">Process</div>'
         '<div style="font-size:0.85rem;opacity:0.7;">AI detects, tracks &amp; classifies students</div>'
         '</div>')
with col3:
    card('<div style="text-align:center;">'
         '<div style="font-size:1.8rem;margin-bottom:0.4rem;">3️⃣</div>'
         '<div style="font-weight:600;margin-bottom:0.2rem;">Results</div>'
         '<div style="font-size:0.85rem;opacity:0.7;">View per-student engagement reports</div>'
         '</div>')

# ── File uploader ─────────────────────────────────────────────────────────

section_header("Select Video", "🎬")

uploaded = st.file_uploader(
    "Choose a video file",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded is not None:
    st.video(uploaded)
    st.caption(f"📁 {uploaded.name} — {uploaded.size / 1024 / 1024:.1f} MB")

    if st.button("🚀 Analyze Engagement", type="primary", use_container_width=True):
        api = get_api_client()

        with st.spinner("Mengunggah video ke server…"):
            try:
                resp = api.upload_video(uploaded, uploaded.name)

            except _requests.exceptions.ReadTimeout:
                # Backend mungkin sudah menerima video tapi response terlambat.
                # Arahkan user ke History daripada tampilkan error teknis.
                st.warning(
                    "⏱️ **Server sedang memproses video Anda.**  \n"
                    "Koneksi ke server timeout, tapi video kemungkinan besar "
                    "sudah diterima dan sedang diproses.  \n\n"
                    "Silakan cek halaman **History** untuk memantau status analisis."
                )
                if st.button("📋 Buka History", type="primary"):
                    st.switch_page("pages/3_History.py")
                st.stop()

            except _requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Tidak bisa terhubung ke server.**  \n"
                    "Pastikan backend sudah berjalan di `http://localhost:8000`."
                )
                st.stop()

            except Exception as e:
                st.error(
                    f"❌ **Upload gagal.**  \n"
                    f"Detail teknis: `{e}`  \n\n"
                    f"Coba lagi atau cek halaman History jika video sudah pernah diupload."
                )
                if st.button("📋 Cek History", key="err_history"):
                    st.switch_page("pages/3_History.py")
                st.stop()

        # Upload berhasil — langsung arahkan ke History untuk polling
        analysis_id = resp["analysis_id"]
        st.success(
            f"✅ **Video berhasil dikirim!**  \n"
            f"Pipeline analisis sedang berjalan di background.  \n"
            f"ID Analisis: `{analysis_id}`"
        )
        st.info("🔄 Anda akan diarahkan ke halaman History untuk memantau progress…")
        time.sleep(2)
        st.session_state["last_analysis_id"] = analysis_id
        st.switch_page("pages/3_History.py")
