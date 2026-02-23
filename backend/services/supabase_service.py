"""
Supabase service — handles Storage (video upload/download) and Database (analyses, student_results).
"""

from __future__ import annotations
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from supabase import create_client, Client

from backend.config import get_settings

settings = get_settings()


def _get_client() -> Client:
    """Create a Supabase client with the service-role key (server-side)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def _get_anon_client() -> Client:
    """Create a Supabase client with the anon key (auth operations)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sign_up(email: str, password: str, full_name: str | None = None) -> dict:
    """Register a new user via Supabase Auth."""
    client = _get_anon_client()
    options = {}
    if full_name:
        options["data"] = {"full_name": full_name}
    res = client.auth.sign_up({"email": email, "password": password, "options": options})

    # When email confirmation is enabled (Supabase default), session is None
    if res.session is not None:
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user_id": res.user.id,
            "email": res.user.email,
        }
    else:
        # User created but needs email confirmation
        return {
            "access_token": None,
            "refresh_token": None,
            "user_id": str(res.user.id) if res.user else None,
            "email": res.user.email if res.user else email,
            "needs_confirmation": True,
        }


def sign_in(email: str, password: str) -> dict:
    """Log in an existing user."""
    client = _get_anon_client()
    res = client.auth.sign_in_with_password({"email": email, "password": password})
    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "user_id": res.user.id,
        "email": res.user.email,
    }


def refresh_session(refresh_token: str) -> dict:
    """Refresh an expired access token."""
    client = _get_anon_client()
    res = client.auth.refresh_session(refresh_token)
    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "user_id": res.user.id,
        "email": res.user.email,
    }


def get_user_from_token(access_token: str) -> dict:
    """Validate a JWT and return the user object."""
    client = _get_anon_client()
    res = client.auth.get_user(access_token)
    return {"user_id": res.user.id, "email": res.user.email}


# ═══════════════════════════════════════════════════════════════════════════════
# STORAGE  (buckets: input-videos, output-videos)
# ═══════════════════════════════════════════════════════════════════════════════

def upload_file(bucket: str, storage_path: str, local_path: str, content_type: str = "video/mp4") -> str:
    """Upload a local file to Supabase Storage. Returns the storage path."""
    client = _get_client()
    with open(local_path, "rb") as f:
        client.storage.from_(bucket).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": content_type},
        )
    return storage_path


def get_signed_url(bucket: str, storage_path: str, expires_in: int = 3600) -> str:
    """Generate a signed (temporary) download URL."""
    client = _get_client()
    res = client.storage.from_(bucket).create_signed_url(storage_path, expires_in)
    return res["signedURL"]


def delete_file(bucket: str, storage_path: str) -> None:
    """Remove a file from Storage."""
    client = _get_client()
    client.storage.from_(bucket).remove([storage_path])


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — analyses table
# ═══════════════════════════════════════════════════════════════════════════════

def create_analysis(user_id: str, original_filename: str, input_video_path: str) -> dict:
    """Insert a new analysis row. Returns the inserted row."""
    client = _get_client()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "original_filename": original_filename,
        "input_video_path": input_video_path,
        "status": "uploading",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = client.table("analyses").insert(row).execute()
    return res.data[0]


def update_analysis(analysis_id: str, **fields) -> dict:
    """Update arbitrary fields on an analysis row."""
    client = _get_client()
    res = client.table("analyses").update(fields).eq("id", analysis_id).execute()
    return res.data[0] if res.data else {}


def get_analysis(analysis_id: str) -> dict | None:
    """Fetch a single analysis by ID."""
    client = _get_client()
    res = client.table("analyses").select("*").eq("id", analysis_id).execute()
    return res.data[0] if res.data else None


def get_user_analyses(user_id: str, limit: int = 50) -> list[dict]:
    """Get all analyses for a user, newest first."""
    client = _get_client()
    res = (
        client.table("analyses")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """Delete an analysis and its student results (cascade via FK or manual)."""
    client = _get_client()
    # Delete student results first
    client.table("student_results").delete().eq("analysis_id", analysis_id).execute()
    # Delete analysis row
    res = (
        client.table("analyses")
        .delete()
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    return len(res.data) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — student_results table
# ═══════════════════════════════════════════════════════════════════════════════

def insert_student_results(analysis_id: str, students: list[dict]) -> None:
    """Bulk-insert per-student majority-vote results."""
    client = _get_client()
    rows = []
    for s in students:
        rows.append({
            "id": str(uuid.uuid4()),
            "analysis_id": analysis_id,
            "track_id": s["track_id"],
            "final_engagement": s["final_engagement"],
            "engaged_votes": s["engaged_votes"],
            "moderate_votes": s["moderate_votes"],
            "disengaged_votes": s["disengaged_votes"],
            "total_frames": s["total_frames"],
            "avg_confidence": round(s["avg_confidence"], 4),
            "vote_percentage": round(s["vote_percentage"], 2),
        })
    if rows:
        client.table("student_results").insert(rows).execute()


def get_student_results(analysis_id: str) -> list[dict]:
    """Fetch per-student results for an analysis."""
    client = _get_client()
    res = (
        client.table("student_results")
        .select("*")
        .eq("analysis_id", analysis_id)
        .order("track_id")
        .execute()
    )
    return res.data
