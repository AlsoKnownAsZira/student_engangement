"""
Video utilities for reading, writing, and processing videos
"""

import cv2
import numpy as np
from pathlib import Path
import os


class VideoReader:
    """Wrapper for video reading with error handling"""
    
    def __init__(self, video_path):
        self.video_path = Path(video_path)  # Convert to Path object
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
    def read(self):
        """Read next frame"""
        return self.cap.read()
    
    def read_frame(self, frame_idx):
        """Read specific frame by index"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return self.cap.read()
    
    def get_frames(self, frame_indices):
        """Get multiple specific frames"""
        frames = []
        for idx in frame_indices:
            ret, frame = self.read_frame(idx)
            if ret:
                frames.append(frame)
        return frames
    
    def release(self):
        """Release video capture"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __str__(self):
        return (f"VideoReader({self.video_path.name})\n"
                f"  Size: {self.width}x{self.height}\n"
                f"  FPS: {self.fps:.2f}\n"
                f"  Frames: {self.total_frames}\n"
                f"  Duration: {self.duration:.2f}s")


class VideoWriter:
    """Wrapper for video writing"""
    
    def __init__(self, output_path, fps, width, height, codec='mp4v'):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")
    
    def write(self, frame):
        """Write frame to video"""
        self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def extract_uniform_frames(video_path, num_frames):
    """
    Extract frames uniformly distributed across video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of tuples (frame_idx, frame)
    """
    with VideoReader(video_path) as reader:
        if num_frames >= reader.total_frames:
            # Extract all frames
            frame_indices = list(range(reader.total_frames))
        else:
            # Uniform sampling
            frame_indices = np.linspace(
                0, 
                reader.total_frames - 1, 
                num_frames, 
                dtype=int
            )
        
        frames = []
        for idx in frame_indices:
            ret, frame = reader.read_frame(idx)
            if ret:
                frames.append((int(idx), frame))
    
    return frames


def extract_random_frames(video_path, num_frames, seed=42):
    """
    Extract random frames from video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples (frame_idx, frame)
    """
    np.random.seed(seed)
    
    with VideoReader(video_path) as reader:
        if num_frames >= reader.total_frames:
            frame_indices = list(range(reader.total_frames))
        else:
            frame_indices = np.random.choice(
                reader.total_frames,
                num_frames,
                replace=False
            )
            frame_indices = sorted(frame_indices)
        
        frames = []
        for idx in frame_indices:
            ret, frame = reader.read_frame(idx)
            if ret:
                frames.append((int(idx), frame))
    
    return frames


def get_video_info(video_path):
    """Get video information as dictionary"""
    try:
        with VideoReader(video_path) as reader:
            return {
                'path': str(video_path),
                'width': reader.width,
                'height': reader.height,
                'fps': reader.fps,
                'total_frames': reader.total_frames,
                'duration': reader.duration
            }
    except Exception as e:
        return {
            'path': str(video_path),
            'error': str(e)
        }


def create_video_grid(frames, grid_size=(2, 2), target_size=(640, 480)):
    """
    Create a grid of frames
    
    Args:
        frames: List of frames
        grid_size: Tuple (rows, cols)
        target_size: Size of each frame in grid
    
    Returns:
        Grid image
    """
    rows, cols = grid_size
    h, w = target_size
    
    # Resize all frames
    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, (w, h)))
    
    # Pad with black frames if needed
    total_cells = rows * cols
    if len(resized) < total_cells:
        black = np.zeros((h, w, 3), dtype=np.uint8)
        resized.extend([black] * (total_cells - len(resized)))
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_frames = resized[i * cols:(i + 1) * cols]
        grid_rows.append(np.hstack(row_frames))
    
    return np.vstack(grid_rows)


def resize_frame(frame, target_size=None, max_size=None):
    """
    Resize frame maintaining aspect ratio
    
    Args:
        frame: Input frame
        target_size: Target (width, height) or None
        max_size: Maximum dimension or None
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if target_size:
        return cv2.resize(frame, target_size)
    
    if max_size:
        scale = min(max_size / w, max_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    return frame


def draw_text_with_background(img, text, position, font_scale=0.6, 
                               thickness=2, bg_color=(0, 0, 0), 
                               text_color=(255, 255, 255), padding=5):
    """Draw text with background rectangle"""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        img, text, (x, y),
        font, font_scale, text_color, thickness
    )


if __name__ == "__main__":
    # Test video utils
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_utils.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Test VideoReader
    print("\n=== Video Info ===")
    info = get_video_info(video_path)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test frame extraction
    print("\n=== Extracting 5 uniform frames ===")
    frames = extract_uniform_frames(video_path, 5)
    print(f"Extracted {len(frames)} frames")
    for idx, frame in frames:
        print(f"  Frame {idx}: shape {frame.shape}")