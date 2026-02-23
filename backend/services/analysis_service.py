"""
Analysis service — majority-voting engagement aggregation and class-level
summary computation from raw per-frame tracking data.
"""

from __future__ import annotations
import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger("analysis_service")

# ── Class-name normalisation ──────────────────────────────────────────────────

OLD_TO_NEW = {
    "high": "engaged",
    "medium": "moderately-engaged",
    "low": "disengaged",
}

VALID_LEVELS = {"engaged", "moderately-engaged", "disengaged"}

# Tie-breaking priority: engaged > moderately-engaged > disengaged
TIE_PRIORITY = ["engaged", "moderately-engaged", "disengaged"]


def _normalise(level: str) -> str:
    """Map old class names (high/medium/low) → new convention."""
    return OLD_TO_NEW.get(level, level)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run majority-vote aggregation on a raw tracking DataFrame.

    Parameters
    ----------
    df : DataFrame with columns
         [frame, track_id, x1, y1, x2, y2,
          detection_conf, engagement_level, engagement_score]

    Returns
    -------
    {
        "students": [
            {
                "track_id": int,
                "final_engagement": str,
                "engaged_votes": int,
                "moderate_votes": int,
                "disengaged_votes": int,
                "total_frames": int,
                "avg_confidence": float,
                "vote_percentage": float,      # % of frames matching final level
            },
            …
        ],
        "class_summary": {
            "total_students": int,
            "total_frames": int,
            "total_detections": int,
            "avg_engagement_score": float,
            "engagement_distribution": {
                "engaged": float,              # fraction 0-1
                "moderately_engaged": float,
                "disengaged": float,
            },
        },
    }
    """
    if df is None or df.empty:
        return _empty_result()

    # Normalise class names
    df = df.copy()
    df["engagement_level"] = df["engagement_level"].apply(_normalise)

    students = _per_student_majority_vote(df)
    class_summary = _class_summary(df, students)

    return {
        "students": students,
        "class_summary": class_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STUDENT MAJORITY VOTING
# ═══════════════════════════════════════════════════════════════════════════════

def _per_student_majority_vote(df: pd.DataFrame) -> list[dict]:
    """
    For each track_id, count occurrences of each engagement level across
    all frames.  The level with the most votes becomes that student's
    **final** engagement level.  Ties are broken by TIE_PRIORITY.
    """
    students: list[dict] = []

    for track_id, group in df.groupby("track_id"):
        counts = group["engagement_level"].value_counts().to_dict()

        engaged_votes    = counts.get("engaged", 0)
        moderate_votes   = counts.get("moderately-engaged", 0)
        disengaged_votes = counts.get("disengaged", 0)
        total_frames     = len(group)
        avg_confidence   = float(group["engagement_score"].mean())

        # Determine winner with tie-breaking
        vote_map = {
            "engaged": engaged_votes,
            "moderately-engaged": moderate_votes,
            "disengaged": disengaged_votes,
        }
        max_votes = max(vote_map.values())
        # Among those tied at max, pick the highest-priority level
        final_engagement = next(
            lv for lv in TIE_PRIORITY if vote_map[lv] == max_votes
        )
        vote_pct = (max_votes / total_frames * 100) if total_frames else 0.0

        students.append({
            "track_id": int(track_id),
            "final_engagement": final_engagement,
            "engaged_votes": engaged_votes,
            "moderate_votes": moderate_votes,
            "disengaged_votes": disengaged_votes,
            "total_frames": total_frames,
            "avg_confidence": round(avg_confidence, 4),
            "vote_percentage": round(vote_pct, 2),
        })

    # Sort by track_id for consistency
    students.sort(key=lambda s: s["track_id"])
    return students


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS-LEVEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _class_summary(df: pd.DataFrame, students: list[dict]) -> dict:
    """
    Aggregate over all students' *final* engagement levels to produce
    a class-level engagement distribution.
    """
    total_students = len(students)
    if total_students == 0:
        return {
            "total_students": 0,
            "total_frames": 0,
            "total_detections": 0,
            "avg_engagement_score": 0.0,
            "engagement_distribution": {
                "engaged": 0.0,
                "moderately_engaged": 0.0,
                "disengaged": 0.0,
            },
        }

    # Distribution based on majority-vote outcomes
    eng_count  = sum(1 for s in students if s["final_engagement"] == "engaged")
    mod_count  = sum(1 for s in students if s["final_engagement"] == "moderately-engaged")
    dis_count  = sum(1 for s in students if s["final_engagement"] == "disengaged")

    return {
        "total_students": total_students,
        "total_frames": int(df["frame"].nunique()),
        "total_detections": len(df),
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 4),
        "engagement_distribution": {
            "engaged": round(eng_count / total_students, 4),
            "moderately_engaged": round(mod_count / total_students, 4),
            "disengaged": round(dis_count / total_students, 4),
        },
    }


def _empty_result() -> dict:
    return {
        "students": [],
        "class_summary": {
            "total_students": 0,
            "total_frames": 0,
            "total_detections": 0,
            "avg_engagement_score": 0.0,
            "engagement_distribution": {
                "engaged": 0.0,
                "moderately_engaged": 0.0,
                "disengaged": 0.0,
            },
        },
    }
