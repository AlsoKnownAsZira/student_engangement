"""
Configuration file for Person Tracking Engagement System
Update paths sesuai dengan struktur folder Anda
"""

import os
from pathlib import Path

# ============================================================================
# PATHS - SESUAIKAN DENGAN SYSTEM ANDA!
# ============================================================================

# Dataset video OUC-CGE
VIDEO_DATASET_ROOT = r"F:\OUC-CGE-dataset"
VIDEO_PATHS = {
    'high': os.path.join(VIDEO_DATASET_ROOT, "high"),
    'med': os.path.join(VIDEO_DATASET_ROOT, "med"),
    'low': os.path.join(VIDEO_DATASET_ROOT, "Low")
}

# Working directory
WORK_DIR = r"D:\kuliah\Skripsi\person-tracking-engagement"

# Output directories
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
POC_RESULTS_DIR = os.path.join(OUTPUT_DIR, "poc_results")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
PERSON_CROPS_DIR = os.path.join(OUTPUT_DIR, "person_crops")
TRAINED_MODELS_DIR = os.path.join(OUTPUT_DIR, "trained_models")

# Runs directory (untuk training results)
RUNS_DIR = r"D:\kuliah\Skripsi\runs"

# Create directories if not exist
for dir_path in [OUTPUT_DIR, POC_RESULTS_DIR, ANNOTATIONS_DIR, 
                 PERSON_CROPS_DIR, TRAINED_MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Detection & Tracking
DETECTION_MODEL = "yolo11s.pt"  #  yolo11n.pt untuk lebih cepat atau yolo11s.pt untuk lebih akurat
POSE_MODEL = "yolo11n-pose.pt"  # untuk pose estimation
TRACKER_CONFIG = "botsort.yaml"  # atau botsort.yaml

# Detection parameters
CONF_THRESHOLD = 0.3  # confidence threshold untuk detection
IOU_THRESHOLD = 0.5   # IoU threshold untuk NMS
PERSON_CLASS_ID = 0   # class ID untuk 'person' di COCO

# Tracking parameters
TRACK_PERSIST = True  # maintain ID across frames
TRACK_BUFFER = 30     # frames to keep lost tracks

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Video processing
PROCESS_FPS = None  # None = original FPS, or set to specific value (e.g., 2)
SKIP_FRAMES = 0     # Skip N frames between processing (0 = process all)
MAX_FRAMES = None   # Maximum frames to process (None = all)

# Image parameters
INPUT_SIZE = 640    # Input size untuk YOLO detection
CROP_SIZE = 224     # Size untuk cropped person images

# ============================================================================
# ENGAGEMENT SCORING PARAMETERS
# ============================================================================
# ============================================================================
# ENGAGEMENT SCORING PARAMETERS (LEGACY - ONLY FOR PHASE 1 POC)
# ============================================================================
# NOTE: Phase 1 menggunakan hardcoded weights di utils/metrics.py
# Phase 2-4 menggunakan ML classifier, TIDAK menggunakan weights ini
# Untuk penggunaan production, lihat Phase 4 (full_pipeline.py)
ENGAGEMENT_WEIGHTS = {
    'pose_upright': 0.35,      # Postur tegak
    'head_forward': 0.30,     # Kepala menghadap depan
    'hands_visible': 0.2,     # Tangan terlihat (menulis/gestur)
    'body_stable': 0.05,      # Tubuh tidak banyak bergerak
    'sitting': 0.1            # Posisi duduk
}

# Engagement thresholds
ENGAGEMENT_THRESHOLDS = {
    'high': 0.7,    # Score >= 0.65 = High
    'medium': 0.4   # 0.5 <= Score < 0.65 = Medium, < 0.5 = Low
}

# Temporal smoothing (untuk mengurangi fluktuasi)
SMOOTHING_WINDOW = 5  # frames

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Colors (BGR format untuk OpenCV)
COLORS = {
    'high': (0, 255, 0),      # Green
    'medium': (0, 165, 255),  # Orange
    'low': (0, 0, 255)        # Red
}

# Visualization settings
BBOX_THICKNESS = 2
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
SHOW_CONFIDENCE = True
SHOW_TRACK_ID = True
SHOW_POSE_KEYPOINTS = True

# ============================================================================
# DATASET PREPARATION PARAMETERS
# ============================================================================

# Frame extraction
FRAMES_PER_VIDEO = 50  # Jumlah sample frames per video
SAMPLING_METHOD = 'uniform'  # 'uniform' or 'random'

# Minimum person size (untuk filter deteksi kecil)
MIN_PERSON_WIDTH = 50   # pixels
MIN_PERSON_HEIGHT = 100 # pixels

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    'model': 'yolo11s-cls.pt',
    'epochs': 50,
    'batch': 32,
    'imgsz': 224,
    'optimizer': 'AdamW',
    'lr0': 0.003,
    'patience': 10,
    'save_period': 5,
    'device': 0,  # GPU 0, or 'cpu'
    'workers': 4,
    'amp': True,  # Automatic Mixed Precision
}

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

VERBOSE = True
SAVE_INTERMEDIATE = True  # Save intermediate results untuk debugging
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_video_list(category=None):
    """
    Get list of video files from dataset
    
    Args:
        category: 'high', 'med', 'low', or None for all
    
    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    if category:
        if category in VIDEO_PATHS:
            folder = VIDEO_PATHS[category]
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        videos.append({
                            'path': os.path.join(folder, file),
                            'category': category,
                            'filename': file
                        })
    else:
        # Get all videos from all categories
        for cat, folder in VIDEO_PATHS.items():
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        videos.append({
                            'path': os.path.join(folder, file),
                            'category': cat,
                            'filename': file
                        })
    
    return videos


def get_engagement_level(score):
    """Convert numeric score to engagement level"""
    if score >= ENGAGEMENT_THRESHOLDS['high']:
        return 'high'
    elif score >= ENGAGEMENT_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'low'


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("PERSON TRACKING ENGAGEMENT - CONFIGURATION")
    print("=" * 80)
    print(f"Video Dataset Root: {VIDEO_DATASET_ROOT}")
    print(f"Working Directory: {WORK_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nDetection Model: {DETECTION_MODEL}")
    print(f"Tracker: {TRACKER_CONFIG}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"\nEngagement Thresholds:")
    print(f"  High: >= {ENGAGEMENT_THRESHOLDS['high']}")
    print(f"  Medium: >= {ENGAGEMENT_THRESHOLDS['medium']}")
    print(f"  Low: < {ENGAGEMENT_THRESHOLDS['medium']}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
    
    # Test: get video list
    videos = get_video_list()
    print(f"\nFound {len(videos)} videos in dataset:")
    for cat in ['high', 'med', 'low']:
        cat_videos = [v for v in videos if v['category'] == cat]
        print(f"  {cat.upper()}: {len(cat_videos)} videos")