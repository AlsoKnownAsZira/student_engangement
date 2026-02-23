"""
History page — browse past analyses.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import streamlit as st
import pandas as pd

from components.auth import require_auth, get_api_client, show_user_sidebar
from fe_config import PAGE_TITLE, PAGE_ICON, ENGAGEMENT_EMOJI, ENGAGEMENT_LABELS

st.set_page_config(page_title=f"History | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
show_user_sidebar()

st.title("Analysis History")

api = get_api_client()

try:
    history = api.get_history()
except Exception as e:
    st.error(f"Failed to load history: {e}")
    st.stop()

analyses = history.get("analyses", [])

if not analyses:
    st.info("You haven't analyzed any videos yet. Go to **Upload** to get started!")
    st.stop()

st.markdown(f"Showing **{len(analyses)}** past analyses.")

# ── Render each analysis as a card ────────────────────────────────────────

for item in analyses:
    status_emoji = {
        "completed": "✅",
        "processing": "⏳",
        "failed": "❌",
        "uploading": "📤",
    }.get(item["status"], "❔")

    with st.expander(
        f"{status_emoji}  **{item['original_filename']}**  —  {item.get('created_at', 'N/A')[:19]}",
        expanded=False,
    ):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Status:** {item['status']}")
        col2.markdown(f"**Students:** {item.get('total_students', '—')}")
        avg = item.get("avg_engagement_score")
        col3.markdown(f"**Avg Score:** {round(avg * 100, 1)}%" if avg else "**Avg Score:** —")

        # Distribution mini-summary
        dist = item.get("engagement_distribution")
        if dist:
            eng = dist.get("engaged", 0) * 100
            mod = dist.get("moderately_engaged", dist.get("moderately-engaged", 0)) * 100
            dis = dist.get("disengaged", 0) * 100
            st.markdown(
                f"{ENGAGEMENT_EMOJI['engaged']} Engaged: **{eng:.0f}%** · "
                f"{ENGAGEMENT_EMOJI['moderately-engaged']} Moderate: **{mod:.0f}%** · "
                f"{ENGAGEMENT_EMOJI['disengaged']} Disengaged: **{dis:.0f}%**"
            )

        # Action buttons
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if item["status"] == "completed":
                if st.button("View Results", key=f"view_{item['analysis_id']}"):
                    st.session_state["last_analysis_id"] = item["analysis_id"]
                    st.switch_page("pages/2_Results.py")
        with bcol2:
            if st.button("Delete", key=f"del_{item['analysis_id']}"):
                try:
                    api.delete_analysis(item["analysis_id"])
                    st.success("Deleted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
