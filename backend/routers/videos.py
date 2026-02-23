"""
Videos router — upload a video and kick off background processing.
"""

from __future__ import annotations
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
import pandas as pd

from backend.config import get_settings
from backend.dependencies import get_current_user
from backend.models.schemas import AnalysisCreate, AnalysisStatusResponse, AnalysisStatus
from backend.services import supabase_service, video_service, analysis_service
from backend.services.pipeline_service import pipeline_manager

settings = get_settings()
router = APIRouter(prefix="/api/videos", tags=["videos"])
logger = logging.getLogger("videos_router")


# ── Upload + process ──────────────────────────────────────────────────────

@router.post("/upload", response_model=AnalysisCreate, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload a classroom video.  Processing starts in the background;
    poll ``/api/videos/{analysis_id}/status`` for progress.
    """
    # Validate extension
    if not video_service.validate_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Validate file size (read into memory check — for streaming, use middleware)
    contents = await file.read()
    if len(contents) > settings.max_video_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max {settings.MAX_VIDEO_SIZE_MB} MB.",
        )

    # Save to temp
    temp_input, uid = video_service.save_temp_upload(
        io.BytesIO(contents), file.filename
    )

    # Upload original to Supabase Storage
    storage_input_path = f"{user['user_id']}/{uid}/input{Path(file.filename).suffix}"
    try:
        supabase_service.upload_file("input-videos", storage_input_path, temp_input)
    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        video_service.cleanup_temp(uid)
        raise HTTPException(status_code=500, detail="Failed to upload video to storage.")

    # Create DB row
    analysis_row = supabase_service.create_analysis(
        user_id=user["user_id"],
        original_filename=file.filename,
        input_video_path=storage_input_path,
    )
    analysis_id = analysis_row["id"]

    # Schedule background processing
    background_tasks.add_task(
        _process_video_task,
        analysis_id=analysis_id,
        user_id=user["user_id"],
        uid=uid,
        temp_input=temp_input,
        original_filename=file.filename,
    )

    return AnalysisCreate(analysis_id=analysis_id, status=AnalysisStatus.PROCESSING)


# ── Status polling ────────────────────────────────────────────────────────

@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_status(analysis_id: str, user: dict = Depends(get_current_user)):
    row = supabase_service.get_analysis(analysis_id)
    if not row or row["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisStatusResponse(
        analysis_id=row["id"],
        status=row["status"],
        error_message=row.get("error_message"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK
# ═══════════════════════════════════════════════════════════════════════════════

async def _process_video_task(
    analysis_id: str,
    user_id: str,
    uid: str,
    temp_input: str,
    original_filename: str,
):
    """
    Heavy-lift background task:
    1. Run ML pipeline
    2. Majority-vote analysis
    3. Upload outputs to Supabase
    4. Persist results to DB
    5. Clean up temp files
    """
    try:
        supabase_service.update_analysis(analysis_id, status="processing")

        # 1. Run pipeline
        temp_output = video_service.get_temp_output_path(uid)
        df, h264_video_path, elapsed = await pipeline_manager.process(
            temp_input, temp_output
        )

        if df is None or df.empty:
            supabase_service.update_analysis(
                analysis_id,
                status="failed",
                error_message="No persons detected in video.",
            )
            return

        # 2. Majority-vote analysis
        report = analysis_service.analyse(df)

        # 3. Save CSV to temp then upload
        csv_temp = str(Path(temp_input).parent / "tracking_data.csv")
        df.to_csv(csv_temp, index=False)

        storage_video_path = f"{user_id}/{uid}/output.mp4"
        storage_csv_path = f"{user_id}/{uid}/tracking_data.csv"

        supabase_service.upload_file("output-videos", storage_video_path, h264_video_path)
        supabase_service.upload_file(
            "output-videos", storage_csv_path, csv_temp, content_type="text/csv"
        )

        # 4. Persist to DB
        class_summary = report["class_summary"]
        supabase_service.update_analysis(
            analysis_id,
            status="completed",
            output_video_path=storage_video_path,
            csv_path=storage_csv_path,
            total_frames=class_summary["total_frames"],
            total_students=class_summary["total_students"],
            avg_engagement_score=class_summary["avg_engagement_score"],
            engagement_distribution=class_summary["engagement_distribution"],
            processing_time_seconds=round(elapsed, 2),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        supabase_service.insert_student_results(analysis_id, report["students"])

        logger.info(
            f"Analysis {analysis_id} completed — "
            f"{class_summary['total_students']} students, "
            f"{elapsed:.1f}s"
        )

    except Exception as e:
        logger.exception(f"Processing failed for {analysis_id}")
        supabase_service.update_analysis(
            analysis_id,
            status="failed",
            error_message=str(e)[:500],
        )
    finally:
        video_service.cleanup_temp(uid)
