"""
Plotly chart helpers for engagement visualisation — dark/light theme aware.
V10: 2-class (engaged / not-engaged).
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import plotly.graph_objects as go
import pandas as pd

from fe_config import ENGAGEMENT_CHART_COLORS, ENGAGEMENT_LABELS
from components.styles import get_chart_colors

# Consistent ordering — engaged first, not-engaged second
_LEVELS = ["engaged", "not-engaged"]


def _theme_layout(fig: go.Figure, title: str = "", height: int = 400, **kwargs) -> go.Figure:
    """Apply consistent dark-theme-aware styling to any Plotly figure."""
    c = get_chart_colors()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=16, color=c["font_color"]),
        ) if title else None,
        plot_bgcolor=c["bg"],
        paper_bgcolor=c["paper_bg"],
        font=dict(family="Inter, sans-serif", size=12, color=c["font_color"]),
        height=height,
        margin=dict(t=60 if title else 20, b=40, l=20, r=20),
        legend=dict(
            font=dict(color=c["font_color"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        **kwargs,
    )
    fig.update_xaxes(
        gridcolor=c["grid"],
        zerolinecolor=c["grid"],
        tickfont=dict(color=c["text"]),
        title_font=dict(color=c["text"]),
    )
    fig.update_yaxes(
        gridcolor=c["grid"],
        zerolinecolor=c["grid"],
        tickfont=dict(color=c["text"]),
        title_font=dict(color=c["text"]),
    )
    return fig


def _norm_distribution(distribution: dict) -> dict[str, float]:
    """Normalise key naming variants -> {'engaged', 'not-engaged'}."""
    return {
        "engaged":     distribution.get("engaged", 0),
        "not-engaged": distribution.get(
            "not_engaged",
            distribution.get("not-engaged", 0),
        ),
    }


def engagement_pie_chart(distribution: dict, title: str = "Engagement Distribution") -> go.Figure:
    """Donut chart from distribution dict (2-class)."""
    c = get_chart_colors()
    norm = _norm_distribution(distribution)

    labels = [ENGAGEMENT_LABELS[lv] for lv in _LEVELS]
    values = [norm[lv] for lv in _LEVELS]
    colors = [ENGAGEMENT_CHART_COLORS[lv] for lv in _LEVELS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(color=c["font_color"], width=0),
        ),
        hole=0.5,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color=c["font_color"], size=12, family="Inter, sans-serif"),
        outsidetextfont=dict(color=c["text"], size=11),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    _theme_layout(fig, title=title, height=380)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11, color=c["font_color"]),
        ),
    )
    return fig


def student_engagement_bar(students: list[dict]) -> go.Figure:
    """Horizontal bar chart — one bar per student coloured by final engagement."""
    c = get_chart_colors()
    if not students:
        fig = go.Figure()
        _theme_layout(fig, title="No student data")
        return fig

    df = pd.DataFrame(students)
    df = df.sort_values("track_id")
    df["label"] = df["track_id"].apply(lambda x: f"Student {x}")
    df["color"] = df["final_engagement"].map(ENGAGEMENT_CHART_COLORS)

    fig = go.Figure(go.Bar(
        y=df["label"],
        x=df["vote_percentage"],
        orientation="h",
        marker_color=df["color"],
        text=df["final_engagement"].map(ENGAGEMENT_LABELS),
        textposition="auto",
        textfont=dict(color="#fff", size=11, family="Inter, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Vote: %{x:.1f}%<br>%{text}<extra></extra>",
    ))
    _theme_layout(
        fig,
        title="Per-Student Engagement (Majority Vote)",
        height=max(300, len(df) * 40 + 100),
    )
    fig.update_layout(
        xaxis_title="Majority Vote %",
        margin=dict(t=60, b=40, l=100, r=20),
    )
    return fig


def vote_breakdown_stacked(students: list[dict]) -> go.Figure:
    """Fallback stacked bar — used when CSV per-frame data is unavailable."""
    if not students:
        fig = go.Figure()
        _theme_layout(fig, title="No student data")
        return fig

    df = pd.DataFrame(students).sort_values("track_id")
    labels = [f"Student {tid}" for tid in df["track_id"]]

    fig = go.Figure()
    for level_key, col in [
        ("engaged",     "engaged_votes"),
        ("not-engaged", "not_engaged_votes"),
    ]:
        fig.add_trace(go.Bar(
            y=labels,
            x=df[col],
            name=ENGAGEMENT_LABELS[level_key],
            orientation="h",
            marker_color=ENGAGEMENT_CHART_COLORS[level_key],
            hovertemplate=f"<b>{ENGAGEMENT_LABELS[level_key]}</b>: %{{x}} frames<extra></extra>",
        ))

    _theme_layout(fig, title="Vote Breakdown per Student", height=max(300, len(df) * 40 + 100), barmode="stack")
    fig.update_layout(xaxis_title="Frames", margin=dict(t=60, b=40, l=100, r=20))
    return fig


def student_frame_line(df_student: pd.DataFrame, track_id: int, threshold: float = 0.170) -> go.Figure:
    """
    Zoned line chart: green zone (above threshold) = engaged, red zone (below) = not-engaged.
    Line moves through zones — higher = more engaged, lower = more not-engaged.
    """
    if df_student.empty:
        fig = go.Figure()
        _theme_layout(fig, title="No data for selected student")
        return fig

    df = df_student.sort_values("frame").reset_index(drop=True)
    y_vals = df["prob_engaged"] if "prob_engaged" in df.columns else df.get("engagement_score", pd.Series([0.5] * len(df)))
    levels = df.get("engagement_level", pd.Series(["engaged"] * len(df)))
    labels_text = [ENGAGEMENT_LABELS.get(lv, lv) for lv in levels]

    x_min = int(df["frame"].min())
    x_max = int(df["frame"].max())

    fig = go.Figure()

    # ── Zona hijau (engaged) ──────────────────────────────────────────────
    fig.add_hrect(
        y0=threshold, y1=1.0,
        fillcolor=ENGAGEMENT_CHART_COLORS["engaged"],
        opacity=0.08, layer="below", line_width=0,
    )
    # ── Zona merah (not-engaged) ──────────────────────────────────────────
    fig.add_hrect(
        y0=0, y1=threshold,
        fillcolor=ENGAGEMENT_CHART_COLORS["not-engaged"],
        opacity=0.08, layer="below", line_width=0,
    )

    # ── Garis pembatas threshold ──────────────────────────────────────────
    fig.add_hline(
        y=threshold,
        line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
    )

    # ── Label zona (kanan chart) ──────────────────────────────────────────
    fig.add_annotation(
        x=x_max, y=(1.0 + threshold) / 2,
        text=f"← {ENGAGEMENT_LABELS['engaged']}",
        showarrow=False,
        font=dict(color=ENGAGEMENT_CHART_COLORS["engaged"], size=11),
        xanchor="right",
    )
    fig.add_annotation(
        x=x_max, y=threshold / 2,
        text=f"← {ENGAGEMENT_LABELS['not-engaged']}",
        showarrow=False,
        font=dict(color=ENGAGEMENT_CHART_COLORS["not-engaged"], size=11),
        xanchor="right",
    )

    # ── Garis keterlibatan ────────────────────────────────────────────────
    # Warna marker mengikuti posisi (di atas/bawah threshold), bukan smoothed label
    raw_colors = [
        ENGAGEMENT_CHART_COLORS["engaged"] if v >= threshold else ENGAGEMENT_CHART_COLORS["not-engaged"]
        for v in y_vals
    ]
    raw_labels = [
        ENGAGEMENT_LABELS["engaged"] if v >= threshold else ENGAGEMENT_LABELS["not-engaged"]
        for v in y_vals
    ]

    fig.add_trace(go.Scatter(
        x=df["frame"],
        y=y_vals,
        mode="lines+markers",
        line=dict(color="rgba(200,200,200,0.8)", width=2),
        marker=dict(color=raw_colors, size=7, line=dict(width=0)),
        hovertemplate="Frame %{x}<br>%{customdata}<extra></extra>",
        customdata=raw_labels,
        showlegend=False,
    ))

    _theme_layout(fig, title=f"Student {track_id} — Keterlibatan per Frame", height=340)
    fig.update_layout(
        xaxis_title="Frame",
        yaxis=dict(range=[0, 1.05], tickvals=[0, threshold, 0.5, 1.0],
                   ticktext=["0", f"batas\n({threshold})", "0.5", "1.0"]),
        yaxis_title="Tingkat Keterlibatan",
        margin=dict(t=60, b=50, l=80, r=20),
    )
    return fig


def engagement_summary_metrics(class_summary: dict) -> dict[str, Any]:
    """Return nicely-formatted values suitable for st.metric calls."""
    dist = class_summary.get("engagement_distribution", {})
    norm = _norm_distribution(dist)
    engaged_pct     = norm["engaged"] * 100
    not_engaged_pct = norm["not-engaged"] * 100

    return {
        "total_students":   class_summary.get("total_students", 0),
        "total_frames":     class_summary.get("total_frames", 0),
        "total_detections": class_summary.get("total_detections", 0),
        "avg_score":        round(class_summary.get("avg_engagement_score", 0) * 100, 1),
        "engaged_pct":      round(engaged_pct, 1),
        "not_engaged_pct":  round(not_engaged_pct, 1),
    }
