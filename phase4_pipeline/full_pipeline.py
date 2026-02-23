"""
Phase 4: Full Pipeline with Trained Model - FIXED VERSION
Handles both old (high/medium/low) and new (engaged/moderately-engaged/disengaged) class names
Includes visualization debugging
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
    """Complete pipeline with trained classifier - FIXED"""
    
    # Support BOTH naming conventions
    COLORS_NEW = {
        'engaged': (0, 255, 0),           # Green
        'moderately-engaged': (0, 165, 255),  # Orange
        'disengaged': (0, 0, 255)         # Red
    }
    
    COLORS_OLD = {
        'high': (0, 255, 0),              # Green
        'medium': (0, 165, 255),          # Orange
        'low': (0, 0, 255)                # Red
    }
    
    # Mapping old to new
    CLASS_MAPPING = {
        'high': 'engaged',
        'medium': 'moderately-engaged',
        'low': 'disengaged'
    }
    
    def __init__(self, 
                 detection_model='yolo11s.pt',
                 classifier_model=None,
                 tracker_config='botsort.yaml',
                 conf_threshold=0.3,
                 iou_threshold=0.5,
                 device=0):
        """Initialize full pipeline"""
        self.logger = setup_logger(self.__class__.__name__)
        
        # Load detection model
        self.logger.info(f"Loading detector: {detection_model}")
        self.detector = YOLO(detection_model)
        
        # Load classifier model
        if classifier_model:
            self.logger.info(f"Loading classifier: {classifier_model}")
            self.classifier = YOLO(classifier_model)
            
            # Detect class naming convention
            if hasattr(self.classifier, 'names'):
                class_names = list(self.classifier.names.values())
                self.logger.info(f"Classifier classes: {class_names}")
                
                # Check which convention is used
                if 'high' in class_names or 'medium' in class_names or 'low' in class_names:
                    self.class_convention = 'old'
                    self.colors = self.COLORS_OLD
                    self.logger.warning("⚠️ Using OLD class names (high/medium/low)")
                    self.logger.warning("   Recommended: Retrain with new names (engaged/moderately-engaged/disengaged)")
                else:
                    self.class_convention = 'new'
                    self.colors = self.COLORS_NEW
                    self.logger.info("✓ Using NEW class names (engaged/moderately-engaged/disengaged)")
        else:
            self.classifier = None
            self.class_convention = 'new'
            self.colors = self.COLORS_NEW
            self.logger.warning("No classifier provided - using dummy predictions")
        
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.metrics = EngagementMetrics()
        self.tracking_data = []
        
        self.logger.info("Pipeline initialized successfully")
        self.logger.info(f"Color scheme: {self.colors}")
    
    def normalize_class_name(self, class_name):
        """Normalize class name to new convention"""
        if self.class_convention == 'old' and class_name in self.CLASS_MAPPING:
            return self.CLASS_MAPPING[class_name]
        return class_name
    
    def process_video(self, video_path, output_path=None, save_csv=True, 
                     show_preview=False, limit_frames=None):
        """Process video with full pipeline"""
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
            # Track persons
            self.logger.info("Starting detection and tracking...")
            results = self.detector.track(
                source=str(video_path),
                stream=True,
                tracker=self.tracker_config,
                classes=[0],  # person class
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
                
                # Process each person
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.id is not None:
                            track_id = int(box.id[0])
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Validate bbox
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            # Ensure bbox is within frame
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            # Crop person
                            person_crop = frame[y1:y2, x1:x2]
                            
                            if person_crop.size == 0:
                                continue
                            
                            # Classify engagement
                            if self.classifier:
                                try:
                                    cls_result = self.classifier(person_crop, verbose=False)[0]
                                    
                                    # Get prediction
                                    pred_class = int(cls_result.probs.top1)
                                    pred_conf = float(cls_result.probs.top1conf)
                                    
                                    # Map class index to label
                                    class_names = cls_result.names
                                    engagement_level = class_names[pred_class]
                                    engagement_score = pred_conf
                                    
                                    # Debug: Log first few predictions
                                    if frame_idx == 0 and track_id <= 3:
                                        self.logger.info(f"Track {track_id}: {engagement_level} ({engagement_score:.3f})")
                                    
                                except Exception as e:
                                    self.logger.warning(f"Classification failed for track {track_id}: {e}")
                                    engagement_level = 'medium' if self.class_convention == 'old' else 'moderately-engaged'
                                    engagement_score = 0.5
                            else:
                                # Dummy prediction
                                engagement_level = 'medium' if self.class_convention == 'old' else 'moderately-engaged'
                                engagement_score = 0.5
                            
                            # Store frame data
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
                
                # Show preview
                if show_preview:
                    cv2.imshow('Pipeline Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Preview stopped by user")
                        break
                
                # Write frame
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
        """Draw person bbox and info with EXPLICIT color debugging"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color - with fallback
        color = self.colors.get(level, (255, 255, 255))  # White if not found
        
        # DEBUG: Force different colors for testing
        if level not in self.colors:
            self.logger.warning(f"⚠️ Level '{level}' not in colors dict! Using white.")
            color = (255, 255, 255)
        
        # Draw bbox with THICK lines for visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw additional border for emphasis
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
        
        # Count levels - support both conventions
        if self.class_convention == 'old':
            level_counts = {'high': 0, 'medium': 0, 'low': 0}
        else:
            level_counts = {'engaged': 0, 'moderately-engaged': 0, 'disengaged': 0}
        
        for score, level in frame_scores.values():
            if level in level_counts:
                level_counts[level] += 1
        
        total = len(frame_scores)
        
        # Summary panel with color indicators
        if self.class_convention == 'old':
            panel_lines = [
                f"Frame: {frame_idx}",
                f"Students: {total}",
                f"High: {level_counts['high']} ({level_counts['high']/total*100:.0f}%)",
                f"Medium: {level_counts['medium']} ({level_counts['medium']/total*100:.0f}%)",
                f"Low: {level_counts['low']} ({level_counts['low']/total*100:.0f}%)"
            ]
        else:
            panel_lines = [
                f"Frame: {frame_idx}",
                f"Students: {total}",
                f"Engaged: {level_counts['engaged']} ({level_counts['engaged']/total*100:.0f}%)",
                f"Moderate: {level_counts['moderately-engaged']} ({level_counts['moderately-engaged']/total*100:.0f}%)",
                f"Disengaged: {level_counts['disengaged']} ({level_counts['disengaged']/total*100:.0f}%)"
            ]
        
        # Draw panel
        panel_h = len(panel_lines) * 35 + 20
        panel_w = 320
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 40
        for i, line in enumerate(panel_lines):
            # Add color indicator for engagement levels
            if i >= 2:  # Skip "Frame" and "Students" lines
                level_key = list(level_counts.keys())[i-2]
                color = self.colors.get(level_key, (255, 255, 255))
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
            'class_convention': self.class_convention
        }
        
        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 4: Full pipeline - FIXED VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--classifier', type=str, required=True, help='Path to trained classifier')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--detector', type=str, default='yolo11s.pt', help='Detection model')
    parser.add_argument('--device', default=0, help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--limit', type=int, default=None, help='Limit frames (testing)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullPipeline(
        detection_model=args.detector,
        classifier_model=args.classifier,
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
        logger.info(f"\nClass Convention: {stats['class_convention'].upper()}")
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