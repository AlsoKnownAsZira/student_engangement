"""
Phase 4: Full Pipeline with Trained Model - 1-MODEL VERSION
Handles Object Detection & Tracking in a single pass.
Maps model class names to standard format: engaged/moderately-engaged/disengaged

v2 Improvements:
  - EngagementSmoother: temporal voting over N frames to stabilize labels
  - Custom BotSORT tracker (custom_botsort.yaml) with tuned thresholds
  - Lower conf_threshold (0.2) to catch small/distant students

v3 Improvements (SAHI mode):
  - Optional SAHI (Slicing Aided Hyper Inference) via --sahi flag
  - Splits each frame into overlapping tiles before detection,
    improving recall for small/distant students (63% of objects <1% frame area)
  - SAHI mode uses supervision ByteTrack as tracker
  - Install deps: pip install sahi supervision
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

import config
from utils.video_utils import VideoReader, VideoWriter, draw_text_with_background
from utils.metrics import EngagementMetrics
from utils.logger import setup_logger

# Optional SAHI + supervision imports (only needed when --sahi flag is used)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import supervision as sv
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


class EngagementSmoother:
    """
    Temporal smoothing for engagement labels.
    Instead of using per-frame raw predictions, this class accumulates
    predictions over a sliding window and returns the confidence-weighted
    majority vote. This prevents label flickering (e.g., engaged → disengaged
    → engaged across consecutive frames).
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(self, track_id, engagement_level, confidence):
        """Record a new observation for a tracked person"""
        self.history[track_id].append((engagement_level, confidence))
    
    def get_smoothed(self, track_id):
        """
        Get the smoothed engagement label via confidence-weighted voting.
        
        Returns:
            (engagement_level, avg_confidence) or (None, 0.0) if no history.
        """
        if track_id not in self.history or len(self.history[track_id]) == 0:
            return None, 0.0
        
        votes = defaultdict(float)
        for level, conf in self.history[track_id]:
            votes[level] += conf  # weight by detection confidence
        
        best_level = max(votes, key=votes.get)
        total_weight = sum(votes.values())
        avg_conf = votes[best_level] / len(self.history[track_id])
        return best_level, avg_conf
    
    def cleanup_stale(self, active_ids):
        """Remove history for IDs no longer being tracked"""
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]


class FullPipeline:
    """1-Model Architecture Pipeline"""
    
    # Standard DB Colors
    COLORS = {
        'engaged': (0, 255, 0),           # Green
        'moderately-engaged': (0, 165, 255),  # Orange
        'disengaged': (0, 0, 255)         # Red
    }
    
    # Mapping various model class conventions to our standard
    CLASS_NORMALIZE = {
        'high': 'engaged',
        'High': 'engaged',
        'engaged': 'engaged',
        
        'medium': 'moderately-engaged',
        'Medium': 'moderately-engaged',
        'moderately-engaged': 'moderately-engaged',
        
        'low': 'disengaged',
        'Low': 'disengaged',
        'disengaged': 'disengaged'
    }
    
    def __init__(self,
                 main_model='models/roboflow_weights.pt',
                 tracker_config='custom_botsort.yaml',
                 conf_threshold=0.2,
                 iou_threshold=0.5,
                 device=0,
                 smoothing_window=10,
                 use_sahi=False,
                 sahi_slice_size=640,
                 sahi_overlap=0.2):
        """Initialize full pipeline with a single object detection model

        Args:
            main_model: Path to YOLO model weights
            tracker_config: Tracker config YAML (default: custom_botsort.yaml with tuned params)
            conf_threshold: Detection confidence threshold (lowered to 0.2 for small objects)
            iou_threshold: IoU threshold for NMS
            device: GPU device ID or 'cpu'
            smoothing_window: Number of frames for temporal engagement smoothing
            use_sahi: Enable SAHI sliced inference for better small-object detection
            sahi_slice_size: Pixel size of each SAHI tile (default: 640)
            sahi_overlap: Overlap ratio between tiles (default: 0.2 = 20%)
        """
        self.logger = setup_logger(self.__class__.__name__)

        # Load main model
        self.logger.info(f"Loading main detector & classifier: {main_model}")
        self.model = YOLO(main_model)

        # Determine classes from model
        if hasattr(self.model, 'names'):
            self.model_class_names = self.model.names
            self.logger.info(f"Model raw classes: {self.model_class_names}")
        else:
            self.logger.warning("Could not detect class names from model!")
            self.model_class_names = {0: 'medium'} # fallback

        # Resolve tracker config to absolute path so it works regardless of CWD
        tracker_path = Path(tracker_config)
        if not tracker_path.is_absolute() and not tracker_path.exists():
            local_path = Path(__file__).parent / tracker_config
            if local_path.exists():
                tracker_config = str(local_path)
                self.logger.info(f"Resolved tracker config: {tracker_config}")
            else:
                self.logger.warning(
                    f"Tracker config '{tracker_config}' not found locally or in "
                    f"{Path(__file__).parent} — Ultralytics will use its own default."
                )
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Temporal smoothing for stable engagement labels
        self.smoother = EngagementSmoother(window_size=smoothing_window)
        self.logger.info(f"Engagement smoother enabled (window={smoothing_window} frames)")

        self.metrics = EngagementMetrics()
        self.tracking_data = []

        # SAHI setup
        self.use_sahi = use_sahi
        if use_sahi:
            if not SAHI_AVAILABLE:
                raise ImportError(
                    "SAHI mode requires extra packages.\n"
                    "Install: pip install sahi supervision"
                )
            device_str = f"cuda:{device}" if isinstance(device, int) else str(device)
            self.logger.info(f"Initializing SAHI model (slice={sahi_slice_size}, overlap={sahi_overlap})")
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=main_model,
                confidence_threshold=conf_threshold,
                device=device_str,
            )
            self.sahi_slice_size = sahi_slice_size
            self.sahi_overlap = sahi_overlap
            self.sv_tracker = sv.ByteTrack()
            self.logger.info("SAHI mode enabled (ByteTrack tracker)")

        mode = "SAHI+ByteTrack" if use_sahi else "standard BotSORT"
        self.logger.info(f"Pipeline initialized successfully (1-Model Architecture v3, {mode})")
    
    def get_normalized_class(self, class_id):
        """Map raw model class to standardized 'engaged/moderately/disengaged'"""
        raw_name = self.model_class_names.get(int(class_id), 'medium')
        return self.CLASS_NORMALIZE.get(raw_name, 'moderately-engaged')
    
    def process_video(self, video_path, output_path=None, save_csv=True,
                     show_preview=False, limit_frames=None):
        """Process video with the single-model pipeline.
        Routes to SAHI mode automatically if use_sahi=True."""
        self.logger.info(f"Processing: {video_path}")

        if self.use_sahi:
            return self._process_video_sahi(
                video_path, output_path=output_path, save_csv=save_csv,
                show_preview=show_preview, limit_frames=limit_frames
            )
        
        # Reset
        self.metrics.reset()
        self.tracking_data = []
        
        # Open video
        reader = VideoReader(video_path)
        self.logger.info(str(reader))
        
        # Setup writer
        writer = None
        if output_path:
            writer = VideoWriter(
                output_path,
                reader.fps,
                reader.width,
                reader.height
            )
            self.logger.info(f"Output will be saved to: {output_path}")
        
        frame_idx = 0
        
        try:
            # Detect and Track in one go
            self.logger.info("Starting SINGLE-PASS detection and tracking...")
            results = self.model.track(
                source=str(video_path),
                stream=True,
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                persist=True,
                verbose=False
            )
            
            for result in results:
                # Check frame limit
                if limit_frames and frame_idx >= limit_frames:
                    self.logger.info(f"Reached frame limit: {limit_frames}")
                    break
                
                frame = result.orig_img.copy()
                frame_scores = {}
                
                # Debug: Log frame processing
                if frame_idx % 30 == 0:
                    self.logger.info(f"Processing frame {frame_idx}...")
                
                untracked_counter = 99000
                
                # Process each detection
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Assign dummy ID if tracker dropped this box
                        if box.id is not None:
                            track_id = int(box.id[0])
                        else:
                            track_id = untracked_counter
                            untracked_counter += 1
                            
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])

                        x1, y1, x2, y2 = map(int, bbox)

                        # Skip detections that are too small to be a real student.
                        # Objects < 0.3% of frame area are likely furniture/bags/noise.
                        frame_area = frame.shape[0] * frame.shape[1]
                        bbox_area = (x2 - x1) * (y2 - y1)
                        if bbox_area / frame_area < 0.003:
                            continue

                        # Standardize classification label
                        raw_engagement = self.get_normalized_class(class_id)
                        
                        # Apply temporal smoothing: accumulate this observation
                        # then use the smoothed (voted) label for display & logging
                        self.smoother.update(track_id, raw_engagement, conf)
                        engagement_level, engagement_score = self.smoother.get_smoothed(track_id)
                        
                        # Fallback if smoother returns None
                        if engagement_level is None:
                            engagement_level = raw_engagement
                            engagement_score = conf
                        
                        # Store frame data for metrics
                        frame_scores[track_id] = (engagement_score, engagement_level)
                        
                        # Log tracking data
                        self.tracking_data.append({
                            'frame': frame_idx,
                            'track_id': track_id,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'detection_conf': conf,
                            'raw_engagement': raw_engagement,
                            'engagement_level': engagement_level,
                            'engagement_score': engagement_score
                        })
                            
                        # Draw on frame
                        self.draw_person(frame, track_id, bbox, 
                                       engagement_level, engagement_score)
                
                # Update metrics
                self.metrics.add_frame(frame_scores)
                
                # Draw summary
                self.draw_summary(frame, frame_scores, frame_idx)
                
                # Show preview (GUI)
                if show_preview:
                    cv2.imshow('Pipeline Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Preview stopped by user")
                        break
                
                # Write frame to output video
                if writer:
                    writer.write(frame)
                
                frame_idx += 1
        
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise
        
        finally:
            reader.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        self.logger.info(f"Processing complete: {frame_idx} frames")
        
        # Create DataFrame
        df = pd.DataFrame(self.tracking_data)
        
        # Save CSV
        if save_csv and output_path and len(df) > 0:
            csv_path = Path(output_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved tracking data: {csv_path}")
        
        return df
    
    def _sahi_to_sv_detections(self, sahi_result):
        """Convert SAHI PredictionResult to supervision Detections."""
        preds = sahi_result.object_prediction_list
        if not preds:
            return sv.Detections.empty()

        xyxy = np.array([
            [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
            for p in preds
        ], dtype=np.float32)
        confs = np.array([p.score.value for p in preds], dtype=np.float32)
        cls_ids = np.array([p.category.id for p in preds], dtype=int)

        return sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls_ids)

    def _process_video_sahi(self, video_path, output_path=None, save_csv=True,
                            show_preview=False, limit_frames=None):
        """SAHI mode: frame-by-frame loop with sliced inference + ByteTrack."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

        writer = None
        if output_path:
            writer = VideoWriter(output_path, fps, width, height)
            self.logger.info(f"Output: {output_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if limit_frames and frame_idx >= limit_frames:
                    self.logger.info(f"Reached frame limit: {limit_frames}")
                    break

                if frame_idx % 30 == 0:
                    self.logger.info(f"SAHI processing frame {frame_idx}/{total}...")

                # --- SAHI sliced detection ---
                sahi_result = get_sliced_prediction(
                    frame,
                    self.sahi_model,
                    slice_height=self.sahi_slice_size,
                    slice_width=self.sahi_slice_size,
                    overlap_height_ratio=self.sahi_overlap,
                    overlap_width_ratio=self.sahi_overlap,
                    verbose=False,
                )
                detections = self._sahi_to_sv_detections(sahi_result)

                # --- ByteTrack update ---
                if len(detections) > 0:
                    tracked = self.sv_tracker.update_with_detections(detections)
                else:
                    tracked = sv.Detections.empty()

                # --- Same logic as standard mode ---
                frame_scores = {}
                untracked_counter = 99000

                for i in range(len(tracked)):
                    track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else untracked_counter
                    if tracked.tracker_id is None:
                        untracked_counter += 1

                    bbox = tracked.xyxy[i]
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                    class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0

                    x1, y1, x2, y2 = map(int, bbox)
                    raw_engagement = self.get_normalized_class(class_id)

                    self.smoother.update(track_id, raw_engagement, conf)
                    engagement_level, engagement_score = self.smoother.get_smoothed(track_id)
                    if engagement_level is None:
                        engagement_level = raw_engagement
                        engagement_score = conf

                    frame_scores[track_id] = (engagement_score, engagement_level)

                    self.tracking_data.append({
                        'frame': frame_idx,
                        'track_id': track_id,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'detection_conf': conf,
                        'raw_engagement': raw_engagement,
                        'engagement_level': engagement_level,
                        'engagement_score': engagement_score,
                    })

                    self.draw_person(frame, track_id, bbox, engagement_level, engagement_score)

                active_ids = set(frame_scores.keys())
                self.smoother.cleanup_stale(active_ids)
                self.metrics.add_frame(frame_scores)
                self.draw_summary(frame, frame_scores, frame_idx)

                if show_preview:
                    cv2.imshow('Pipeline Preview (SAHI)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if writer:
                    writer.write(frame)

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        self.logger.info(f"SAHI processing complete: {frame_idx} frames")
        df = pd.DataFrame(self.tracking_data)

        if save_csv and output_path and len(df) > 0:
            csv_path = Path(output_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved tracking data: {csv_path}")

        return df

    def draw_person(self, frame, track_id, bbox, level, score):
        """Draw person bbox and info"""
        x1, y1, x2, y2 = map(int, bbox)
        
        color = self.COLORS.get(level, (255, 255, 255))
        
        # Draw bbox with THICK lines for visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, 1)
        
        # Label with shortened level name
        level_short = level.replace('moderately-', 'mod-').replace('medium', 'med')
        label = f"ID:{track_id} | {level_short.upper()}"
        label2 = f"Conf: {score:.2f}"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (label2_w, label2_h), _ = cv2.getTextSize(label2, font, font_scale, thickness)
        
        # Draw filled rectangle for label
        cv2.rectangle(frame, 
                     (x1, y1 - label_h - label2_h - 15), 
                     (x1 + max(label_w, label2_w) + 10, y1),
                     color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1 + 5, y1 - label2_h - 10),
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, label2, (x1 + 5, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    def draw_summary(self, frame, frame_scores, frame_idx):
        """Draw engagement summary panel"""
        if not frame_scores:
            return
        
        level_counts = {'engaged': 0, 'moderately-engaged': 0, 'disengaged': 0}
        
        for score, level in frame_scores.values():
            if level in level_counts:
                level_counts[level] += 1
        
        total = len(frame_scores)
        
        panel_lines = [
            f"Frame: {frame_idx}",
            f"Students: {total}",
            f"Engaged: {level_counts['engaged']} ({level_counts['engaged']/total*100:.0f}%)" if total > 0 else "Engaged: 0",
            f"Moderate: {level_counts['moderately-engaged']} ({level_counts['moderately-engaged']/total*100:.0f}%)" if total > 0 else "Moderate: 0",
            f"Disengaged: {level_counts['disengaged']} ({level_counts['disengaged']/total*100:.0f}%)" if total > 0 else "Disengaged: 0"
        ]
        
        # Draw panel
        panel_h = len(panel_lines) * 35 + 20
        panel_w = 320
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 40
        for i, line in enumerate(panel_lines):
            if i >= 2:  # Skip "Frame" and "Students" lines
                level_key = list(level_counts.keys())[i-2]
                color = self.COLORS.get(level_key, (255, 255, 255))
                cv2.circle(frame, (25, y-8), 8, color, -1)
                cv2.putText(frame, line, (45, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, line, (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35
    
    def get_statistics(self):
        """Get detailed statistics from tracking data"""
        if not self.tracking_data:
            return None
        
        df = pd.DataFrame(self.tracking_data)
        
        stats = {
            'total_frames': df['frame'].nunique(),
            'total_detections': len(df),
            'unique_students': df['track_id'].nunique(),
            'engagement_distribution': df['engagement_level'].value_counts().to_dict(),
            'avg_confidence': df['engagement_score'].mean(),
            'avg_students_per_frame': len(df) / df['frame'].nunique(),
            'model_type': 'single-model-object-detection'
        }
        
        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 4: Full pipeline - 1-MODEL VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to detection/classification model')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--device', default=0, help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--conf', type=float, default=0.2, help='Detection confidence threshold (default: 0.2 for small objects)')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--limit', type=int, default=None, help='Limit frames (testing)')
    parser.add_argument('--smoothing', type=int, default=10, help='Temporal smoothing window size in frames (default: 10)')
    parser.add_argument('--tracker', type=str, default='custom_botsort.yaml', help='Tracker config file')
    parser.add_argument('--sahi', action='store_true', help='Enable SAHI sliced inference (better for small/distant students)')
    parser.add_argument('--slice-size', type=int, default=640, help='SAHI tile size in pixels (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.2, help='SAHI tile overlap ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullPipeline(
        main_model=args.model,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
        device=args.device,
        smoothing_window=args.smoothing,
        use_sahi=args.sahi,
        sahi_slice_size=args.slice_size,
        sahi_overlap=args.overlap,
    )
    
    # Process video
    df = pipeline.process_video(
        args.video,
        output_path=args.output,
        save_csv=True,
        show_preview=args.preview,
        limit_frames=args.limit
    )
    
    # Print summary
    logger = setup_logger("main")
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    stats = pipeline.get_statistics()
    if stats:
        logger.info(f"Total Frames: {stats['total_frames']}")
        logger.info(f"Total Detections: {stats['total_detections']}")
        logger.info(f"Unique Students: {stats['unique_students']}")
        logger.info(f"Avg Students/Frame: {stats['avg_students_per_frame']:.1f}")
        logger.info(f"Avg Confidence: {stats['avg_confidence']:.3f}")
        
        logger.info(f"\nEngagement Distribution:")
        for level, count in stats['engagement_distribution'].items():
            pct = count / stats['total_detections'] * 100
            logger.info(f"  {level}: {count} ({pct:.1f}%)")
    
    logger.info(f"\nOutput saved:")
    logger.info(f"  Video: {args.output}")
    logger.info(f"  CSV: {Path(args.output).with_suffix('.csv')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()