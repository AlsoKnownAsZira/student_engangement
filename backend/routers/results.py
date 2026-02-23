"""
Results router — fetch analysis results, download CSV, get video URL, history.
"""

from __future__ import annotations
import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_current_user
from backend.models.schemas import (
    AnalysisHistoryResponse,
    AnalysisHistoryItem,
    AnalysisResultResponse,
    AnalysisStatus,
    ClassSummary,
    EngagementDistribution,
    StudentResult,
)
from backend.services import supabase_service

router = APIRouter(prefix="/api/results", tags=["results"])
logger = logging.getLogger("results_router")


# ── Single analysis result ────────────────────────────────────────────────

@router.get("/{analysis_id}", response_model=AnalysisResultResponse)
async def get_result(analysis_id: str, user: dict = Depends(get_current_user)):
    row = supabase_service.get_analysis(analysis_id)
    if not row or row["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Build student list
    student_rows = supabase_service.get_student_results(analysis_id)
    students = [StudentResult(**s) for s in student_rows]

    # Engagement distribution
    dist_raw = row.get("engagement_distribution") or {}
    dist = EngagementDistribution(
        engaged=dist_raw.get("engaged", 0),
        moderately_engaged=dist_raw.get("moderately_engaged", 0),
        disengaged=dist_raw.get("disengaged", 0),
    )

    class_summary = ClassSummary(
        total_students=row.get("total_students") or 0,
        total_frames=row.get("total_frames") or 0,
        total_detections=sum(s.total_frames for s in students),
        avg_engagement_score=row.get("avg_engagement_score") or 0.0,
        engagement_distribution=dist,
    )

    # Signed URLs
    output_video_url = None
    csv_url = None
    if row.get("output_video_path"):
        try:
            output_video_url = supabase_service.get_signed_url(
                "output-videos", row["output_video_path"]
            )
        except Exception:
            pass
    if row.get("csv_path"):
        try:
            csv_url = supabase_service.get_signed_url(
                "output-videos", row["csv_path"]
            )
        except Exception:
            pass

    return AnalysisResultResponse(
        analysis_id=row["id"],
        original_filename=row["original_filename"],
        status=row["status"],
        created_at=row.get("created_at"),
        completed_at=row.get("completed_at"),
        processing_time_seconds=row.get("processing_time_seconds"),
        class_summary=class_summary,
        students=students,
        output_video_url=output_video_url,
        csv_download_url=csv_url,
    )


# ── Download CSV (redirect to signed URL) ────────────────────────────────

@router.get("/{analysis_id}/csv")
async def download_csv(analysis_id: str, user: dict = Depends(get_current_user)):
    row = supabase_service.get_analysis(analysis_id)
    if not row or row["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if not row.get("csv_path"):
        raise HTTPException(status_code=404, detail="CSV not available yet")

    url = supabase_service.get_signed_url("output-videos", row["csv_path"])
    return {"csv_download_url": url}


# ── Get annotated video URL ──────────────────────────────────────────────

@router.get("/{analysis_id}/video")
async def get_video_url(analysis_id: str, user: dict = Depends(get_current_user)):
    row = supabase_service.get_analysis(analysis_id)
    if not row or row["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if not row.get("output_video_path"):
        raise HTTPException(status_code=404, detail="Video not available yet")

    url = supabase_service.get_signed_url("output-videos", row["output_video_path"])
    return {"output_video_url": url}


# ── History ───────────────────────────────────────────────────────────────

@router.get("/", response_model=AnalysisHistoryResponse)
async def list_history(user: dict = Depends(get_current_user)):
    rows = supabase_service.get_user_analyses(user["user_id"])
    items = []
    for r in rows:
        dist_raw = r.get("engagement_distribution") or {}
        items.append(
            AnalysisHistoryItem(
                analysis_id=r["id"],
                original_filename=r["original_filename"],
                status=r["status"],
                created_at=r.get("created_at"),
                completed_at=r.get("completed_at"),
                total_students=r.get("total_students"),
                avg_engagement_score=r.get("avg_engagement_score"),
                engagement_distribution=EngagementDistribution(
                    engaged=dist_raw.get("engaged", 0),
                    moderately_engaged=dist_raw.get("moderately_engaged", 0),
                    disengaged=dist_raw.get("disengaged", 0),
                ) if dist_raw else None,
            )
        )
    return AnalysisHistoryResponse(total=len(items), analyses=items)


# ── Delete ────────────────────────────────────────────────────────────────

@router.delete("/{analysis_id}")
async def delete_analysis(analysis_id: str, user: dict = Depends(get_current_user)):
    row = supabase_service.get_analysis(analysis_id)
    if not row or row["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Remove storage files
    for bucket, key in [
        ("input-videos", row.get("input_video_path")),
        ("output-videos", row.get("output_video_path")),
        ("output-videos", row.get("csv_path")),
    ]:
        if key:
            try:
                supabase_service.delete_file(bucket, key)
            except Exception:
                pass

    supabase_service.delete_analysis(analysis_id, user["user_id"])
    return {"detail": "Deleted"}
