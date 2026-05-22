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
from i18n import t

st.set_page_config(page_title=f"Results | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

# ── Determine which analysis to show ──────────────────────────────────────

analysis_id = st.session_state.get("last_analysis_id")

if not analysis_id:
    hero_section(
        title=t("results_no_analysis"),
        subtitle=t("results_no_analysis_sub"),
        emoji="📊",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("btn_go_upload"), type="primary", use_container_width=True):
            st.switch_page("pages/1_Upload.py")
    with col2:
        if st.button(t("btn_go_history"), use_container_width=True):
            st.switch_page("pages/3_History.py")
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────

api = get_api_client()

try:
    result = api.get_result(analysis_id)
except Exception as e:
    st.error(t("results_load_err", e))
    st.stop()

if result["status"] != "completed":
    st.warning(t("results_not_ready", result["status"]))
    st.stop()

class_summary = result["class_summary"]
students = result["students"]
metrics = engagement_summary_metrics(class_summary)
p = _palette()

# ── Hero ──────────────────────────────────────────────────────────────────

hero_section(
    title=t("results_title"),
    subtitle=f"Video: {result['original_filename']}",
    emoji="📊",
)

# ── Header metrics ────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric(t("metric_students"), metrics["total_students"])
col2.metric(t("metric_frames"), metrics["total_frames"])
col3.metric(t("metric_confidence"), f"{metrics['avg_score']}%")
if result.get("processing_time_seconds"):
    col4.metric(t("metric_time"), f"{result['processing_time_seconds']:.1f}s")
else:
    col4.metric(t("metric_detections"), metrics["total_detections"])

# ── Inference speed metrics (shown only when timing data is available) ────

det_ms = result.get("avg_detector_ms")
cls_ms = result.get("avg_classifier_ms")
pipe_ms = result.get("avg_pipeline_ms_per_frame")

if pipe_ms is not None:
    section_header(t("section_inference_speed"), "⚡")
    icol1, icol2, icol3, icol4 = st.columns(4)
    icol1.metric(t("metric_detector_ms"), f"{det_ms:.1f} ms" if det_ms is not None else "—")
    icol2.metric(t("metric_classifier_ms"), f"{cls_ms:.1f} ms" if cls_ms is not None else "—")
    icol3.metric(t("metric_pipeline_ms"), f"{pipe_ms:.1f} ms")
    eff_fps = round(1000 / pipe_ms, 1) if pipe_ms > 0 else 0.0
    icol4.metric(t("metric_eff_fps"), f"{eff_fps} fps")

st.divider()

# ── Engagement distribution overview ──────────────────────────────────────

section_header(t("section_class_summary"), "📈")

col_pie, col_bars = st.columns([1, 1])

with col_pie:
    fig_pie = engagement_pie_chart(class_summary.get("engagement_distribution", {}))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bars:
    for level, emoji_icon, pct_key, color in [
        ("engaged", ENGAGEMENT_EMOJI["engaged"], "engaged_pct", ENGAGEMENT_COLORS["engaged"]),
        ("not-engaged", ENGAGEMENT_EMOJI["not-engaged"], "not_engaged_pct", ENGAGEMENT_COLORS["not-engaged"]),
    ]:
        pct_val = metrics[pct_key]
        bar_bg = "rgba(128,128,128,0.15)"
        label = ENGAGEMENT_LABELS[level]
        inner_html = (
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.4rem;">{emoji_icon}</span>'
            f'<div style="flex:1;">'
            f'<div style="font-weight:600;color:var(--text-color);font-family:Inter,sans-serif;">{label}</div>'
            f'<div style="margin-top:4px;height:8px;border-radius:4px;background:{bar_bg};overflow:hidden;">'
            f'<div style="width:{min(pct_val, 100)}%;height:100%;background:{color};border-radius:4px;transition:width 0.5s ease;"></div>'
            f'</div></div>'
            f'<span style="font-weight:700;font-size:1.1rem;color:{color};font-family:Inter,sans-serif;">{pct_val}%</span>'
            f'</div>'
        )
        card(inner_html)

    st.markdown(
        f'<p style="color:var(--text-color);opacity:0.8;font-size:0.88rem;margin-top:0.5rem;">'
        f'{t("majority_vote_note", metrics["total_students"])}</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Annotated video ───────────────────────────────────────────────────────

section_header(t("section_video"), "🎬")
show_video(result.get("output_video_url"))

st.divider()

# ── Per-student results ───────────────────────────────────────────────────

section_header(t("section_per_student"), "👥")

tab_table, tab_bar, tab_stack = st.tabs([t("tab_table"), t("tab_bar"), t("tab_stack")])

with tab_table:
    if students:
        df = pd.DataFrame(students)
        df["final_engagement"] = df["final_engagement"].map(
            lambda x: f"{ENGAGEMENT_EMOJI.get(x, '')} {t('label_engaged') if x == 'engaged' else t('label_not_engaged')}"
        )
        df = df.rename(columns={
            "track_id": t("col_student_id"),
            "final_engagement": t("col_engagement"),
            "engaged_votes": t("col_engaged_votes"),
            "not_engaged_votes": t("col_not_engaged_votes"),
            "total_frames": t("col_total_frames"),
            "avg_confidence": t("col_avg_conf"),
            "vote_percentage": t("col_majority_vote"),
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(t("no_student_data"))

with tab_bar:
    fig_bar = student_engagement_bar(students)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_stack:
    csv_url = result.get("csv_download_url")
    if csv_url:
        @st.cache_data(ttl=300, show_spinner=False)
        def _load_csv(url: str) -> pd.DataFrame:
            return pd.read_csv(url)

        try:
            df_csv = _load_csv(csv_url)
            track_ids = sorted(df_csv["track_id"].unique().tolist())
            selected = st.selectbox(
                t("select_student"),
                options=track_ids,
                format_func=lambda x: f"Student {x}",
            )
            import os
            _thr = float(os.environ.get("CLASSIFY_THRESHOLD", 0.170))
            df_sel = df_csv[df_csv["track_id"] == selected]
            from components.charts import student_frame_line
            fig_line = student_frame_line(df_sel, selected, threshold=_thr)
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception:
            fig_stack = vote_breakdown_stacked(students)
            st.plotly_chart(fig_stack, use_container_width=True)
    else:
        fig_stack = vote_breakdown_stacked(students)
        st.plotly_chart(fig_stack, use_container_width=True)

st.divider()

# ── Downloads ─────────────────────────────────────────────────────────────

section_header(t("section_downloads"), "💾")

dcol1, dcol2 = st.columns(2)

with dcol1:
    csv_url = result.get("csv_download_url")
    if csv_url:
        card(
            f'<a href="{csv_url}" target="_blank" style="text-decoration:none;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.6rem;">📄</span>'
            f'<div><div style="font-weight:700;color:var(--primary-color);font-family:Inter,sans-serif;">{t("download_csv_title")}</div>'
            f'<div style="font-size:0.8rem;color:var(--text-color);opacity:0.8;">{t("download_csv_sub")}</div></div>'
            f'</a>'
        )
    else:
        st.info(t("no_csv"))

with dcol2:
    video_url = result.get("output_video_url")
    if video_url:
        card(
            f'<a href="{video_url}" target="_blank" style="text-decoration:none;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:1.6rem;">🎬</span>'
            f'<div><div style="font-weight:700;color:var(--primary-color);font-family:Inter,sans-serif;">{t("download_video_title")}</div>'
            f'<div style="font-size:0.8rem;color:var(--text-color);opacity:0.8;">{t("download_video_sub")}</div></div>'
            f'</a>'
        )
    else:
        st.info(t("no_video"))

st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    if st.button(t("btn_back_history"), use_container_width=True):
        st.switch_page("pages/3_History.py")
