"""
Results page — display analysis output: video, charts, per-student table.
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
    engagement_pie_chart,
    student_engagement_bar,
    vote_breakdown_stacked,
    engagement_summary_metrics,
)
from components.video_player import show_video
from fe_config import (
    PAGE_TITLE,
    PAGE_ICON,
    ENGAGEMENT_COLORS,
    ENGAGEMENT_LABELS,
    ENGAGEMENT_EMOJI,
)

st.set_page_config(page_title=f"Results | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
show_user_sidebar()

st.title("Analysis Results")

# ── Determine which analysis to show ──────────────────────────────────────

analysis_id = st.session_state.get("last_analysis_id")
custom_id = st.text_input("Or enter an analysis ID:", value=analysis_id or "")
if custom_id:
    analysis_id = custom_id

if not analysis_id:
    st.info("No analysis selected. Upload a video or enter an analysis ID above.")
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

# ── Header metrics ────────────────────────────────────────────────────────

st.subheader(f"Video: {result['original_filename']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", metrics["total_students"])
col2.metric("Total Frames", metrics["total_frames"])
col3.metric("Avg Confidence", f"{metrics['avg_score']}%")
if result.get("processing_time_seconds"):
    col4.metric("Processing Time", f"{result['processing_time_seconds']:.1f}s")
else:
    col4.metric("Total Detections", metrics["total_detections"])

st.divider()

# ── Engagement distribution overview ──────────────────────────────────────

st.subheader("Class Engagement Overview")

col_pie, col_bars = st.columns([1, 1])

with col_pie:
    fig_pie = engagement_pie_chart(class_summary.get("engagement_distribution", {}))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bars:
    ecol1, ecol2, ecol3 = st.columns(3)
    ecol1.metric(
        f"{ENGAGEMENT_EMOJI['engaged']} Engaged",
        f"{metrics['engaged_pct']}%",
    )
    ecol2.metric(
        f"{ENGAGEMENT_EMOJI['moderately-engaged']} Moderate",
        f"{metrics['moderate_pct']}%",
    )
    ecol3.metric(
        f"{ENGAGEMENT_EMOJI['disengaged']} Disengaged",
        f"{metrics['disengaged_pct']}%",
    )

    st.markdown("---")
    st.markdown(
        f"Based on **majority voting** across all frames for each of "
        f"the **{metrics['total_students']}** tracked students."
    )

st.divider()

# ── Annotated video ───────────────────────────────────────────────────────

show_video(result.get("output_video_url"))

st.divider()

# ── Per-student results ───────────────────────────────────────────────────

st.subheader("Per-Student Engagement")

tab_table, tab_bar, tab_stack = st.tabs(["Table", "Bar Chart", "Vote Breakdown"])

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

st.subheader("Downloads")

dcol1, dcol2 = st.columns(2)

with dcol1:
    if result.get("csv_download_url"):
        st.markdown(f"[Download Raw CSV]({result['csv_download_url']})")
    else:
        st.info("CSV not available.")

with dcol2:
    if result.get("output_video_url"):
        st.markdown(f"[Download Annotated Video]({result['output_video_url']})")
    else:
        st.info("Annotated video not available.")
