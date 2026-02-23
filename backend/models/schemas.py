"""
Pydantic schemas for request / response models.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


# ── Enums ────────────────────────────────────────────────────────────────────

class EngagementLevel(str, Enum):
    ENGAGED = "engaged"
    MODERATELY_ENGAGED = "moderately-engaged"
    DISENGAGED = "disengaged"


class AnalysisStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Auth ─────────────────────────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_id: Optional[str] = None
    email: str
    needs_confirmation: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Analysis ─────────────────────────────────────────────────────────────────

class AnalysisCreate(BaseModel):
    """Returned right after upload starts."""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AnalysisStatus = AnalysisStatus.UPLOADING


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress_message: Optional[str] = None
    error_message: Optional[str] = None


class StudentResult(BaseModel):
    track_id: int
    final_engagement: EngagementLevel
    engaged_votes: int = 0
    moderate_votes: int = 0
    disengaged_votes: int = 0
    total_frames: int = 0
    avg_confidence: float = 0.0
    vote_percentage: float = 0.0


class EngagementDistribution(BaseModel):
    engaged: float = 0.0
    moderately_engaged: float = 0.0
    disengaged: float = 0.0


class ClassSummary(BaseModel):
    total_students: int = 0
    total_frames: int = 0
    total_detections: int = 0
    avg_engagement_score: float = 0.0
    engagement_distribution: EngagementDistribution = EngagementDistribution()


class AnalysisResultResponse(BaseModel):
    analysis_id: str
    original_filename: str
    status: AnalysisStatus
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    class_summary: ClassSummary = ClassSummary()
    students: list[StudentResult] = []

    output_video_url: Optional[str] = None
    csv_download_url: Optional[str] = None


class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    original_filename: str
    status: AnalysisStatus
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_students: Optional[int] = None
    avg_engagement_score: Optional[float] = None
    engagement_distribution: Optional[EngagementDistribution] = None


class AnalysisHistoryResponse(BaseModel):
    total: int
    analyses: list[AnalysisHistoryItem]
