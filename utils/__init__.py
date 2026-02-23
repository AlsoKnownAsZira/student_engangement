"""
Utility modules for person tracking engagement system
"""

from .video_utils import (
    VideoReader,
    VideoWriter,
    extract_uniform_frames,
    extract_random_frames,
    get_video_info,
    create_video_grid,
    resize_frame,
    draw_text_with_background
)

from .metrics import (
    EngagementScorer,
    EngagementMetrics
)

from .logger import (
    setup_logger,
    ExperimentLogger,
    ProgressLogger
)

__all__ = [
    'VideoReader',
    'VideoWriter',
    'extract_uniform_frames',
    'extract_random_frames',
    'get_video_info',
    'create_video_grid',
    'resize_frame',
    'draw_text_with_background',
    'EngagementScorer',
    'EngagementMetrics',
    'setup_logger',
    'ExperimentLogger',
    'ProgressLogger'
]