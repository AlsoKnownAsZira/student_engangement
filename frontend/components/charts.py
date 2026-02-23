"""
Plotly chart helpers for engagement visualisation — dark/light theme aware.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from fe_config import ENGAGEMENT_CHART_COLORS, ENGAGEMENT_LABELS
from components.styles import get_chart_colors

# Consistent ordering
_LEVELS = ["engaged", "moderately-engaged", "disengaged"]


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


def engagement_pie_chart(distribution: dict, title: str = "Engagement Distribution") -> go.Figure:
    """Donut chart from distribution dict."""
    c = get_chart_colors()

    # Normalise key names (underscores vs dashes)
    norm = {
        "engaged": distribution.get("engaged", 0),
        "moderately-engaged": distribution.get("moderately_engaged", distribution.get("moderately-engaged", 0)),
        "disengaged": distribution.get("disengaged", 0),
    }

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
    """Stacked horizontal bar showing vote counts per student."""
    c = get_chart_colors()
    if not students:
        fig = go.Figure()
        _theme_layout(fig, title="No student data")
        return fig

    df = pd.DataFrame(students).sort_values("track_id")
    labels = [f"Student {tid}" for tid in df["track_id"]]

    fig = go.Figure()
    for level_key, col in [
        ("engaged", "engaged_votes"),
        ("moderately-engaged", "moderate_votes"),
        ("disengaged", "disengaged_votes"),
    ]:
        fig.add_trace(go.Bar(
            y=labels,
            x=df[col],
            name=ENGAGEMENT_LABELS[level_key],
            orientation="h",
            marker_color=ENGAGEMENT_CHART_COLORS[level_key],
            hovertemplate=f"<b>{ENGAGEMENT_LABELS[level_key]}</b>: %{{x}} frames<extra></extra>",
        ))

    _theme_layout(
        fig,
        title="Frame-level Vote Breakdown per Student",
        height=max(300, len(df) * 40 + 100),
        barmode="stack",
    )
    fig.update_layout(
        xaxis_title="Number of Frames",
        margin=dict(t=60, b=40, l=100, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig


def engagement_summary_metrics(class_summary: dict) -> dict[str, Any]:
    """Return nicely-formatted values suitable for st.metric calls."""
    dist = class_summary.get("engagement_distribution", {})
    engaged_pct = dist.get("engaged", 0) * 100
    moderate_pct = dist.get("moderately_engaged", dist.get("moderately-engaged", 0)) * 100
    disengaged_pct = dist.get("disengaged", 0) * 100

    return {
        "total_students": class_summary.get("total_students", 0),
        "total_frames": class_summary.get("total_frames", 0),
        "total_detections": class_summary.get("total_detections", 0),
        "avg_score": round(class_summary.get("avg_engagement_score", 0) * 100, 1),
        "engaged_pct": round(engaged_pct, 1),
        "moderate_pct": round(moderate_pct, 1),
        "disengaged_pct": round(disengaged_pct, 1),
    }
