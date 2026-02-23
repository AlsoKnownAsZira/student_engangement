"""
Plotly chart helpers for engagement visualisation.
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

from fe_config import ENGAGEMENT_COLORS, ENGAGEMENT_LABELS

# Consistent ordering
_LEVELS = ["engaged", "moderately-engaged", "disengaged"]


def engagement_pie_chart(distribution: dict, title: str = "Class Engagement Distribution") -> go.Figure:
    """
    Pie chart from distribution dict like
    {"engaged": 0.65, "moderately_engaged": 0.25, "disengaged": 0.10}
    """
    # Normalise key names (underscores vs dashes)
    norm = {
        "engaged": distribution.get("engaged", distribution.get("engaged", 0)),
        "moderately-engaged": distribution.get("moderately_engaged", distribution.get("moderately-engaged", 0)),
        "disengaged": distribution.get("disengaged", 0),
    }

    labels = [ENGAGEMENT_LABELS[lv] for lv in _LEVELS]
    values = [norm[lv] for lv in _LEVELS]
    colors = [ENGAGEMENT_COLORS[lv] for lv in _LEVELS]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def student_engagement_bar(students: list[dict]) -> go.Figure:
    """
    Horizontal bar chart — one bar per student coloured by final engagement.
    """
    if not students:
        return go.Figure()

    df = pd.DataFrame(students)
    df = df.sort_values("track_id")
    df["label"] = df["track_id"].apply(lambda x: f"Student {x}")
    df["color"] = df["final_engagement"].map(ENGAGEMENT_COLORS)

    fig = go.Figure(go.Bar(
        y=df["label"],
        x=df["vote_percentage"],
        orientation="h",
        marker_color=df["color"],
        text=df["final_engagement"].map(ENGAGEMENT_LABELS),
        textposition="auto",
    ))
    fig.update_layout(
        title="Per-Student Engagement (Majority Vote)",
        xaxis_title="Majority Vote %",
        yaxis_title="",
        height=max(300, len(df) * 40 + 100),
        margin=dict(t=60, b=40, l=100, r=20),
    )
    return fig


def vote_breakdown_stacked(students: list[dict]) -> go.Figure:
    """
    Stacked horizontal bar showing engaged / moderate / disengaged
    frame counts per student.
    """
    if not students:
        return go.Figure()

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
            marker_color=ENGAGEMENT_COLORS[level_key],
        ))

    fig.update_layout(
        barmode="stack",
        title="Frame-level Vote Breakdown per Student",
        xaxis_title="Number of Frames",
        yaxis_title="",
        height=max(300, len(df) * 40 + 100),
        margin=dict(t=60, b=40, l=100, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def engagement_summary_metrics(class_summary: dict) -> dict[str, Any]:
    """
    Return nicely-formatted values suitable for st.metric calls.
    """
    dist = class_summary.get("engagement_distribution", {})
    engaged_pct = dist.get("engaged", dist.get("engaged", 0)) * 100
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
