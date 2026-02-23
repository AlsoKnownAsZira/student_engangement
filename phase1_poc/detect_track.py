"""
Phase 1 POC: Person Detection and Tracking
Detect and track persons in video with engagement scoring
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict

from utils.metrics import EngagementScorer, EngagementMetrics
from utils.video_utils import VideoReader, VideoWriter
from utils.logger import setup_logger


class PersonTracker:
    """Person detection and tracking with engagement scoring"""
    
    def __init__(self, 
                 detection_model='yolo11s.pt',
                 pose_model='yolo11n-pose.pt',
                 tracker_config='botsort.yaml',
                 conf_threshold=0.3,
                 device=0):
        """
        Initialize person tracker
        
        Args:
            detection_model: Path to detection model
            pose_model: Path to pose estimation model
            tracker_config: Tracker configuration
            conf_threshold: Confidence threshold
            device: Device (0 for GPU, 'cpu' for CPU)
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        # Load models
        self.logger.info(f"Loading detection model: {detection_model}")
        self.detector = YOLO(detection_model)
        
        self.logger.info(f"Loading pose model: {pose_model}")
        self.pose_model = YOLO(pose_model)
        
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Engagement scorer
        self.engagement_scorer = EngagementScorer()
        
        # Metrics
        self.metrics = EngagementMetrics()
        
        # Tracking data
        self.tracking_data = []
        
        self.logger.info("Person tracker initialized")
    
    def process_video(self, video_path, output_path=None, save_video=True):
        """
        Process video with person detection, tracking, and engagement scoring
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_video: Whether to save output video
        
        Returns:
            DataFrame with tracking results
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Reset metrics
        self.metrics.reset()
        self.tracking_data = []
        
        # Read video
        reader = VideoReader(video_path)
        self.logger.info(str(reader))
        
        # Setup video writer if needed
        writer = None
        if save_video and output_path:
            writer = VideoWriter(
                output_path,
                reader.fps,
                reader.width,
                reader.height
            )
            self.logger.info(f"Saving output to: {output_path}")
        
        frame_idx = 0
        
        try:
            # Process with tracking
            results = self.detector.track(
                source=str(video_path),
                stream=True,
                tracker=self.tracker_config,
                classes=[0],  # person only
                conf=self.conf_threshold,
                device=self.device,
                persist=True,
                verbose=False
            )
            
            for result in results:
                frame = result.orig_img
                frame_scores = {}
                
                # Process each detected person
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.id is not None:
                            track_id = int(box.id[0])
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            # Crop person region
                            x1, y1, x2, y2 = map(int, bbox)
                            person_crop = frame[y1:y2, x1:x2]
                            
                            # Pose estimation
                            pose_result = self.pose_model(
                                person_crop,
                                verbose=False
                            )[0]
                            
                            # Get keypoints
                            keypoints = None
                            if (pose_result.keypoints is not None and 
                                len(pose_result.keypoints.data) > 0):
                                keypoints = pose_result.keypoints.data[0].cpu().numpy()
                            
                            # Calculate engagement score
                            engagement_score = self.engagement_scorer.calculate_score(
                                keypoints,
                                track_id=track_id
                            )
                            engagement_level = self.engagement_scorer.get_engagement_level(
                                engagement_score
                            )
                            
                            # Store results
                            frame_scores[track_id] = (engagement_score, engagement_level)
                            
                            # Log tracking data
                            self.tracking_data.append({
                                'frame': frame_idx,
                                'track_id': track_id,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': conf,
                                'engagement_score': engagement_score,
                                'engagement_level': engagement_level
                            })
                
                # Update metrics
                self.metrics.add_frame(frame_scores)
                
                # Write frame (will be visualized later)
                if writer:
                    writer.write(frame)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 30 == 0:
                    self.logger.info(f"Processed {frame_idx} frames...")
        
        finally:
            reader.release()
            if writer:
                writer.release()
        
        self.logger.info(f"Processing complete: {frame_idx} frames")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.tracking_data)
        
        return df
    
    def get_summary(self):
        """Get engagement summary statistics"""
        class_summary = self.metrics.get_class_summary()
        students_summary = self.metrics.get_all_students_summary()
        
        return {
            'class': class_summary,
            'students': students_summary
        }
    
    def save_results(self, output_dir, video_name):
        """
        Save tracking results to files
        
        Args:
            output_dir: Output directory
            video_name: Video name (for filenames)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tracking data
        if len(self.tracking_data) > 0:
            df = pd.DataFrame(self.tracking_data)
            csv_path = output_dir / f"{video_name}_tracking.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved tracking data: {csv_path}")
        
        # Save summary
        summary = self.get_summary()
        
        # Class summary
        class_summary = pd.DataFrame([summary['class']])
        class_path = output_dir / f"{video_name}_class_summary.csv"
        class_summary.to_csv(class_path, index=False)
        self.logger.info(f"Saved class summary: {class_path}")
        
        # Student summary
        if summary['students']:
            students_summary = pd.DataFrame(list(summary['students'].values()))
            students_path = output_dir / f"{video_name}_students_summary.csv"
            students_summary.to_csv(students_path, index=False)
            self.logger.info(f"Saved students summary: {students_path}")


def test_single_video(video_path, output_dir):
    """Test tracker on single video"""
    logger = setup_logger("test_poc")
    
    logger.info("=" * 80)
    logger.info("PHASE 1 POC - Person Tracking Test")
    logger.info("=" * 80)
    
    # Initialize tracker
    tracker = PersonTracker(
        detection_model='yolo11s.pt',
        pose_model='yolo11n-pose.pt',
        conf_threshold=0.3,
        device=0
    )
    
    # Process video
    video_name = Path(video_path).stem
    output_video = Path(output_dir) / f"{video_name}_tracked.mp4"
    
    df = tracker.process_video(
        video_path,
        output_path=output_video,
        save_video=True
    )
    
    # Save results
    tracker.save_results(output_dir, video_name)
    
    # Print summary
    summary = tracker.get_summary()
    
    logger.info("\n" + "=" * 80)
    logger.info("CLASS ENGAGEMENT SUMMARY")
    logger.info("=" * 80)
    
    class_summary = summary['class']
    logger.info(f"Total observations: {class_summary['total_observations']}")
    logger.info(f"Number of students: {class_summary['num_students']}")
    logger.info(f"Mean engagement score: {class_summary['mean_score']:.3f}")
    logger.info(f"Median engagement score: {class_summary['median_score']:.3f}")
    
    logger.info("\nEngagement Distribution:")
    for level, pct in class_summary['level_percentages'].items():
        logger.info(f"  {level.upper()}: {pct:.1f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("PER-STUDENT ENGAGEMENT")
    logger.info("=" * 80)
    
    for student_summary in summary['students'].values():
        logger.info(f"\nStudent ID: {student_summary['track_id']}")
        logger.info(f"  Mean score: {student_summary['mean_score']:.3f}")
        logger.info(f"  Dominant level: {student_summary['dominant_level'].upper()}")
        logger.info(f"  Observations: {student_summary['total_observations']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("POC Test Complete!")
    logger.info("=" * 80)
    
    return df, summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python detect_track.py <video_path> <output_dir>")
        print("Example: python detect_track.py video.mp4 ../outputs/poc_results")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    test_single_video(video_path, output_dir)