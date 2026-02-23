"""
Results page — display analysis output: video, charts, per-student table.
Directly fetches results from session state — no manual ID input needed.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import streamlit as st
import pandas as pd

from components.auth import require_auth, get_api_client, show_user_sidebar
from components.charts import (
    engagement_pie_chart, student_engagement_bar,
    vote_breakdown_stacked, engagement_summary_metrics,
)
from components.video_player import show_video
from components.styles import inject_global_css, hero_section, section_header, card, init_theme, _palette
from fe_config import (
    PAGE_TITLE, PAGE_ICON,
    ENGAGEMENT_COLORS, ENGAGEMENT_LABELS, ENGAGEMENT_EMOJI,
)

st.set_page_config(page_title=f"Results | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

# ── Determine which analysis to show ──────────────────────────────────────

analysis_id = st.session_state.get("last_analysis_id")

if not analysis_id:
    hero_section(
        title="No Analysis Selected",
        subtitle="Upload a video or select an analysis from History to view results.",
        emoji="📊",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Go to Upload", type="primary", use_container_width=True):
            st.switch_page("pages/1_Upload.py")
    with col2:
        if st.button("📋 Go to History", use_container_width=True):
            st.switch_page("pages/3_History.py")
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────

api = get_api_client()

try:
    result = api.get_result(analysis_id)
except Exception as e:
    st.error(f"Could not load results: {e}")
    st.stop()

if result["status"] != "completed":
    st.warning(f"Analysis status: **{result['status']}**. Results are not ready yet.")
    st.stop()

class_summary = result["class_summary"]
students = result["students"]
metrics = engagement_summary_metrics(class_summary)
p = _palette()

# ── Hero ──────────────────────────────────────────────────────────────────

hero_section(
    title="Analysis Results",
    subtitle=f"Video: {result['original_filename']}",
    emoji="📊",
)

# ── Header metrics ────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Students", metrics["total_students"])
col2.metric("🎞️ Total Frames", metrics["total_frames"])
col3.metric("🎯 Avg Confidence", f"{metrics['avg_score']}%")
if result.get("processing_time_seconds"):
    col4.metric("⏱️ Processing Time", f"{result['processing_time_seconds']:.1f}s")
else:
    col4.metric("📊 Total Detections", metrics["total_detections"])

st.divider()

# ── Engagement distribution overview ──────────────────────────────────────

section_header("Class Engagement Overview", "📈")

col_pie, col_bars = st.columns([1, 1])

with col_pie:
    fig_pie = engagement_pie_chart(class_summary.get("engagement_distribution", {}))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bars:
    for level, emoji_icon, pct_key, color in [
        ("engaged", ENGAGEMENT_EMOJI["engaged"], "engaged_pct", ENGAGEMENT_COLORS["engaged"]),
        ("moderately-engaged", ENGAGEMENT_EMOJI["moderately-engaged"], "moderate_pct", ENGAGEMENT_COLORS["moderately-engaged"]),
        ("disengaged", ENGAGEMENT_EMOJI["disengaged"], "disengaged_pct", ENGAGEMENT_COLORS["disengaged"]),
    ]:
        pct_val = metrics[pct_key]
        bar_bg = p['bg_secondary']
        label = ENGAGEMENT_LABELS[level]
        inner_html = (
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.4rem;">{emoji_icon}</span>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:600;color:{p["text_primary"]} !important;font-family:Inter,sans-serif;">{label}</div>'
            f'<div style="margin-top:4px;height:8px;border-radius:4px;background:{bar_bg};overflow:hidden;">'
            f'<div style="width:{min(pct_val, 100)}%;height:100%;background:{color};border-radius:4px;transition:width 0.5s ease;"></div>'
            f'</div></div>'
            f'<span style="font-weight:700;font-size:1.1rem;color:{color} !important;font-family:Inter,sans-serif;">{pct_val}%</span>'
            f'</div>'
        )
        card(inner_html)

    st.markdown(
        f'<p style="color:{p["text_secondary"]} !important;font-size:0.88rem;margin-top:0.5rem;">'
        f'Based on <b>majority voting</b> across all frames for each of the '
        f'<b>{metrics["total_students"]}</b> tracked students.</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Annotated video ───────────────────────────────────────────────────────

section_header("Annotated Video", "🎬")
show_video(result.get("output_video_url"))

st.divider()

# ── Per-student results ───────────────────────────────────────────────────

section_header("Per-Student Engagement", "👥")

tab_table, tab_bar, tab_stack = st.tabs(["📋 Table", "📊 Bar Chart", "📈 Vote Breakdown"])

with tab_table:
    if students:
        df = pd.DataFrame(students)
        df["final_engagement"] = df["final_engagement"].map(
            lambda x: f"{ENGAGEMENT_EMOJI.get(x, '')} {ENGAGEMENT_LABELS.get(x, x)}"
        )
        df = df.rename(columns={
            "track_id": "Student ID",
            "final_engagement": "Engagement Level",
            "engaged_votes": "Engaged Frames",
            "moderate_votes": "Moderate Frames",
            "disengaged_votes": "Disengaged Frames",
            "total_frames": "Total Frames",
            "avg_confidence": "Avg Confidence",
            "vote_percentage": "Majority Vote %",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No student data available.")

with tab_bar:
    fig_bar = student_engagement_bar(students)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_stack:
    fig_stack = vote_breakdown_stacked(students)
    st.plotly_chart(fig_stack, use_container_width=True)

st.divider()

# ── Downloads ─────────────────────────────────────────────────────────────

section_header("Downloads", "💾")

dcol1, dcol2 = st.columns(2)

with dcol1:
    csv_url = result.get("csv_download_url")
    if csv_url:
        card(
            f'<a href="{csv_url}" target="_blank" style="text-decoration:none;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:1.5rem;">📄</span>'
            f'<div><div style="font-weight:600;color:{p["accent"]} !important;font-family:Inter,sans-serif;">Download Raw CSV</div>'
            f'<div style="font-size:0.8rem;color:{p["text_secondary"]} !important;">Per-frame tracking data</div></div>'
            f'</a>'
        )
    else:
        st.info("CSV not available.")

with dcol2:
    video_url = result.get("output_video_url")
    if video_url:
        card(
            f'<a href="{video_url}" target="_blank" style="text-decoration:none;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:1.5rem;">🎬</span>'
            f'<div><div style="font-weight:600;color:{p["accent"]} !important;font-family:Inter,sans-serif;">Download Annotated Video</div>'
            f'<div style="font-size:0.8rem;color:{p["text_secondary"]} !important;">Video with bounding boxes &amp; labels</div></div>'
            f'</a>'
        )
    else:
        st.info("Annotated video not available.")

st.markdown("")
if st.button("← Back to History", use_container_width=True):
    st.switch_page("pages/3_History.py")
