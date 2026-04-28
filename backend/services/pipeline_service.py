"""
Pipeline service — wrapper around the V10 2-stage TwoStagePipeline
that makes it usable from a web context (temp files, re-encoding, etc.).
"""

from __future__ import annotations
import subprocess
import shutil
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Patch the project so imports inside phase4_pipeline work properly.
# TwoStagePipeline does `import config` and `from utils.…`, which expect the
# project root to be on sys.path.
# ---------------------------------------------------------------------------
import sys

from backend.config import get_settings

settings = get_settings()
_project_root = settings.PROJECT_ROOT
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from phase4_pipeline.full_pipeline import TwoStagePipeline   # noqa: E402

logger = logging.getLogger("pipeline_service")


class PipelineManager:
    """
    Singleton-ish manager that loads detector + classifier once and provides
    a ``process()`` helper for each request.
    """

    def __init__(self):
        self._pipeline: Optional[TwoStagePipeline] = None
        self._lock = None  # asyncio.Lock — set in load_models

    # ── lifecycle ─────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load YOLO detector + classifier into memory. Called at FastAPI startup."""
        import asyncio

        self._lock = asyncio.Lock()

        logger.info("Loading V10 2-stage models …")
        start = time.time()
        self._pipeline = TwoStagePipeline(
            detector_model=settings.detection_model_abs,
            classifier_model=settings.classifier_model_abs,
            classify_threshold=settings.CLASSIFY_THRESHOLD,
            tracker_config=settings.TRACKER_CONFIG,
            conf_threshold=settings.CONF_THRESHOLD,
            iou_threshold=settings.IOU_THRESHOLD,
            device=settings.resolved_device,
            smoothing_window=settings.SMOOTHING_WINDOW,
            frame_stride=settings.FRAME_STRIDE,
            classifier_imgsz=settings.CLASSIFIER_IMGSZ,
        )
        elapsed = time.time() - start
        logger.info(
            f"Models loaded in {elapsed:.1f}s "
            f"(device={settings.resolved_device}, "
            f"stride={settings.FRAME_STRIDE}, "
            f"thr={settings.CLASSIFY_THRESHOLD})"
        )

    def is_ready(self) -> bool:
        return self._pipeline is not None

    # ── processing ────────────────────────────────────────────────────────

    async def process(
        self,
        input_video: str | Path,
        output_video: str | Path,
    ) -> tuple[pd.DataFrame, str, float]:
        """
        Run the V10 detection + tracking + classification pipeline.

        Returns
        -------
        df          : pd.DataFrame — raw per-frame tracking data
        final_video : str          — path to browser-playable (H.264) video
        elapsed     : float        — wall-clock seconds
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded yet — call load_models() first.")

        # Only one video at a time (GPU memory / model state)
        async with self._lock:
            start = time.time()

            input_video = str(input_video)
            output_video = str(output_video)

            import asyncio
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                self._run_pipeline,
                input_video,
                output_video,
            )

            h264_path = self._reencode_h264(output_video)

            elapsed = time.time() - start
            return df, h264_path, elapsed

    def _run_pipeline(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Blocking call — executed inside a thread pool."""
        df = self._pipeline.process_video(
            video_path=input_path,
            output_path=output_path,
            save_csv=False,       # videos.py saves CSV from df itself
            show_preview=False,
        )
        return df

    # ── H.264 re-encoding ─────────────────────────────────────────────────

    @staticmethod
    def _reencode_h264(mp4v_path: str) -> str:
        """
        OpenCV writes mp4v codec which browsers cannot play.
        Re-encode to H.264 with ffmpeg.  Returns the path to the H.264 file.
        If ffmpeg is not available, returns the original path as-is.
        """
        src = Path(mp4v_path)
        dst = src.with_name(src.stem + "_h264" + src.suffix)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found — skipping H.264 re-encode. "
                           "Video may not play in browser.")
            return str(src)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            "-an",                  # drop audio (classroom video)
            str(dst),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            src.unlink(missing_ok=True)
            logger.info(f"H.264 re-encode complete → {dst}")
            return str(dst)
        except Exception as e:
            logger.warning(f"H.264 re-encode failed ({e}); using mp4v fallback.")
            return str(src)


# Module-level singleton
pipeline_manager = PipelineManager()
