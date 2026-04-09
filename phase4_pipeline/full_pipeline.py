"""
Phase 4: Full Pipeline with Trained Model - 1-MODEL VERSION
Handles Object Detection & Tracking in a single pass.
Maps model class names to standard format: engaged/moderately-engaged/disengaged
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from ultralytics import YOLO

import config
from utils.video_utils import VideoReader, VideoWriter, draw_text_with_background
from utils.metrics import EngagementMetrics
from utils.logger import setup_logger


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
                 tracker_config='botsort.yaml',
                 conf_threshold=0.3, # Adjust to optimal
                 iou_threshold=0.5,
                 device=0):
        """Initialize full pipeline with a single object detection model"""
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
            
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.metrics = EngagementMetrics()
        self.tracking_data = []
        
        self.logger.info("Pipeline initialized successfully (1-Model Architecture)")
    
    def get_normalized_class(self, class_id):
        """Map raw model class to standardized 'engaged/moderately/disengaged'"""
        raw_name = self.model_class_names.get(int(class_id), 'medium')
        return self.CLASS_NORMALIZE.get(raw_name, 'moderately-engaged')
    
    def process_video(self, video_path, output_path=None, save_csv=True, 
                     show_preview=False, limit_frames=None):
        """Process video with the single-model pipeline"""
        self.logger.info(f"Processing: {video_path}")
        
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
                        
                        # Standardize classification label
                        engagement_level = self.get_normalized_class(class_id)
                        engagement_score = conf # The detection confidence represents the class confidence
                        
                        # Store frame data for metrics
                        frame_scores[track_id] = (engagement_score, engagement_level)
                        
                        # Log tracking data
                        self.tracking_data.append({
                            'frame': frame_idx,
                            'track_id': track_id,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'detection_conf': conf,
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
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--limit', type=int, default=None, help='Limit frames (testing)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullPipeline(
        main_model=args.model,
        conf_threshold=args.conf,
        device=args.device
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