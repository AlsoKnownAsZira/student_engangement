"""
Analysis service — majority-voting engagement aggregation and class-level
summary computation from raw per-frame tracking data.

V10 schema: 2-class only — `engaged` / `not-engaged`.
"""

from __future__ import annotations
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("analysis_service")

# ── Class-name normalisation ──────────────────────────────────────────────────

# Map any legacy / 3-class label to the 2-class V10 schema.
LEGACY_TO_NEW = {
    "high":               "engaged",
    "engaged":            "engaged",
    "medium":             "engaged",            # V5 / V10 merge: med -> engaged
    "moderately-engaged": "engaged",
    "low":                "not-engaged",
    "disengaged":         "not-engaged",
    "not-engaged":        "not-engaged",
    "notengaged":         "not-engaged",
}

VALID_LEVELS = {"engaged", "not-engaged"}

# Tie-breaking priority — engaged wins ties (rare in 2-class)
TIE_PRIORITY = ["engaged", "not-engaged"]


def _normalise(level: str) -> str:
    return LEGACY_TO_NEW.get(level, level)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run majority-vote aggregation on a raw tracking DataFrame.

    Returns
    -------
    {
        "students": [
            {
                "track_id": int,
                "final_engagement": "engaged" | "not-engaged",
                "engaged_votes": int,
                "not_engaged_votes": int,
                "total_frames": int,
                "avg_confidence": float,
                "vote_percentage": float,
            }, ...
        ],
        "class_summary": {
            "total_students": int,
            "total_frames": int,
            "total_detections": int,
            "avg_engagement_score": float,
            "engagement_distribution": {
                "engaged": float,        # fraction 0-1
                "not_engaged": float,
            },
        },
    }
    """
    if df is None or df.empty:
        return _empty_result()

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
    students: list[dict] = []

    for track_id, group in df.groupby("track_id"):
        counts = group["engagement_level"].value_counts().to_dict()

        engaged_votes     = int(counts.get("engaged", 0))
        not_engaged_votes = int(counts.get("not-engaged", 0))
        total_frames      = len(group)
        avg_confidence    = float(group["engagement_score"].mean())

        vote_map = {
            "engaged":     engaged_votes,
            "not-engaged": not_engaged_votes,
        }
        max_votes = max(vote_map.values())
        final_engagement = next(
            lv for lv in TIE_PRIORITY if vote_map[lv] == max_votes
        )
        vote_pct = (max_votes / total_frames * 100) if total_frames else 0.0

        students.append({
            "track_id": int(track_id),
            "final_engagement": final_engagement,
            "engaged_votes": engaged_votes,
            "not_engaged_votes": not_engaged_votes,
            "total_frames": total_frames,
            "avg_confidence": round(avg_confidence, 4),
            "vote_percentage": round(vote_pct, 2),
        })

    students.sort(key=lambda s: s["track_id"])
    return students


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS-LEVEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _class_summary(df: pd.DataFrame, students: list[dict]) -> dict:
    unique_track_count = len(students)
    if unique_track_count == 0:
        return {
            "total_students": 0,
            "total_frames": 0,
            "total_detections": 0,
            "avg_engagement_score": 0.0,
            "engagement_distribution": {
                "engaged": 0.0,
                "not_engaged": 0.0,
            },
        }

    # Peak concurrent: max unique IDs visible in any single frame.
    # Robust against ID fragmentation — a re-ID'd student gets a new ID over
    # time but still occupies only one bbox per frame, so the per-frame count
    # stays correct even when the cumulative unique-ID count is inflated.
    peak_concurrent = int(df.groupby("frame")["track_id"].nunique().max())
    total_students = peak_concurrent

    eng_count = sum(1 for s in students if s["final_engagement"] == "engaged")
    ne_count  = unique_track_count - eng_count

    return {
        "total_students":   total_students,
        "total_frames":     int(df["frame"].nunique()),
        "total_detections": len(df),
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 4),
        "engagement_distribution": {
            "engaged":     round(eng_count / unique_track_count, 4),
            "not_engaged": round(ne_count / unique_track_count, 4),
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
                "not_engaged": 0.0,
            },
        },
    }
