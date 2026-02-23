"""
History page — browse past analyses with styled cards.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import streamlit as st

from components.auth import require_auth, get_api_client, show_user_sidebar
from components.styles import (
    inject_global_css, hero_section, section_header,
    status_badge, init_theme, _palette,
)
from fe_config import (
    PAGE_TITLE, PAGE_ICON,
    ENGAGEMENT_EMOJI, ENGAGEMENT_LABELS, ENGAGEMENT_COLORS,
)

st.set_page_config(page_title=f"History | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

hero_section(
    title="Analysis History",
    subtitle="Browse and manage all your past analyses",
    emoji="📋",
)

api = get_api_client()
p = _palette()

try:
    history = api.get_history()
except Exception as e:
    st.error(f"Failed to load history: {e}")
    st.stop()

analyses = history.get("analyses", [])

if not analyses:
    empty_style = (
        f"text-align:center;padding:3rem 1rem;background:{p['bg_card']};"
        f"border:1px solid {p['border']};border-radius:14px;"
    )
    st.markdown(
        f'<div style="{empty_style}">'
        f'<div style="font-size:3rem;margin-bottom:0.8rem;">📭</div>'
        f'<div style="font-size:1.1rem;font-weight:600;color:{p["text_primary"]} !important;margin-bottom:0.4rem;">No analyses yet</div>'
        f'<div style="color:{p["text_secondary"]} !important;font-size:0.9rem;">Upload a classroom video to get started!</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    if st.button("📤 Go to Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_Upload.py")
    st.stop()

st.markdown(
    f'<p style="color:{p["text_secondary"]} !important;">Showing <b>{len(analyses)}</b> past analyses</p>',
    unsafe_allow_html=True,
)

# ── Render each analysis ──────────────────────────────────────────────────

for item in analyses:
    status = item["status"]
    filename = item["original_filename"]
    created = item.get("created_at", "N/A")[:16].replace("T", " ")
    total_students = item.get("total_students", "—")
    avg = item.get("avg_engagement_score")
    avg_display = f"{round(avg * 100, 1)}%" if avg else "—"
    dist = item.get("engagement_distribution")

    # Build distribution dots HTML
    dist_html = ""
    if dist and status == "completed":
        eng = dist.get("engaged", 0) * 100
        mod = dist.get("moderately_engaged", dist.get("moderately-engaged", 0)) * 100
        dis = dist.get("disengaged", 0) * 100
        dot_style = "width:10px;height:10px;border-radius:50%;"
        dist_html = (
            f'<div style="display:flex;gap:16px;margin-top:10px;flex-wrap:wrap;">'
            f'<div style="display:flex;align-items:center;gap:4px;"><div style="{dot_style}background:{ENGAGEMENT_COLORS["engaged"]};"></div><span style="font-size:0.82rem;color:{p["text_primary"]} !important;">Engaged <b>{eng:.0f}%</b></span></div>'
            f'<div style="display:flex;align-items:center;gap:4px;"><div style="{dot_style}background:{ENGAGEMENT_COLORS["moderately-engaged"]};"></div><span style="font-size:0.82rem;color:{p["text_primary"]} !important;">Moderate <b>{mod:.0f}%</b></span></div>'
            f'<div style="display:flex;align-items:center;gap:4px;"><div style="{dot_style}background:{ENGAGEMENT_COLORS["disengaged"]};"></div><span style="font-size:0.82rem;color:{p["text_primary"]} !important;">Disengaged <b>{dis:.0f}%</b></span></div>'
            f'</div>'
        )

    badge = status_badge(status)
    card_style = (
        f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:14px;"
        f"padding:1.2rem 1.5rem;margin-bottom:0.8rem;box-shadow:0 2px 8px {p['shadow']};"
    )
    meta_style = f"display:flex;gap:24px;margin-top:8px;color:{p['text_secondary']} !important;font-size:0.85rem;"
    title_style = f"font-weight:700;font-size:1.05rem;color:{p['text_primary']} !important;font-family:Inter,sans-serif;"

    st.markdown(
        f'<div style="{card_style}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<span style="{title_style}">🎬 {filename}</span>'
        f'{badge}'
        f'</div>'
        f'<div style="{meta_style}"><span style="color:{p["text_secondary"]} !important;">🕐 {created}</span><span style="color:{p["text_secondary"]} !important;">👥 {total_students} students</span><span style="color:{p["text_secondary"]} !important;">🎯 {avg_display}</span></div>'
        f'{dist_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Action buttons
    bcol1, bcol2, bcol3 = st.columns([2, 2, 1])
    with bcol1:
        if status == "completed":
            if st.button("📊 View Results", key=f"view_{item['analysis_id']}", type="primary", use_container_width=True):
                st.session_state["last_analysis_id"] = item["analysis_id"]
                st.switch_page("pages/2_Results.py")
    with bcol3:
        if st.button("🗑️", key=f"del_{item['analysis_id']}", help="Delete this analysis"):
            try:
                api.delete_analysis(item["analysis_id"])
                st.success("Deleted!")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
