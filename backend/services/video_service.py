"""
Video service — temp-file management, validation, cleanup.
"""

from __future__ import annotations
import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import BinaryIO

from backend.config import get_settings

settings = get_settings()
logger = logging.getLogger("video_service")


def validate_extension(filename: str) -> bool:
    """Check that the file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in settings.allowed_extensions_list


def save_temp_upload(file: BinaryIO, original_filename: str) -> tuple[str, str]:
    """
    Write an uploaded file to a temp directory.

    Returns
    -------
    (temp_input_path, unique_id)
    """
    uid = uuid.uuid4().hex[:12]
    ext = Path(original_filename).suffix
    temp_dir = settings.temp_path / uid
    temp_dir.mkdir(parents=True, exist_ok=True)

    dest = temp_dir / f"input{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file, f)

    logger.info(f"Saved temp upload → {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(dest), uid


def get_temp_output_path(uid: str) -> str:
    """Return a temp path for the annotated output video."""
    temp_dir = settings.temp_path / uid
    temp_dir.mkdir(parents=True, exist_ok=True)
    return str(temp_dir / "output.mp4")


def cleanup_temp(uid: str) -> None:
    """Remove the temp folder for a given processing run."""
    temp_dir = settings.temp_path / uid
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir: {temp_dir}")
