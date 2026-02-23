"""
Phase 1 POC: Visualization
Visualize tracking results with bboxes, IDs, and engagement levels
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from utils.video_utils import VideoReader, VideoWriter, draw_text_with_background
from utils.logger import setup_logger


class TrackingVisualizer:
    """Visualize person tracking and engagement results"""
    
    # Colors for engagement levels (BGR)
    COLORS = {
        'high': (0, 255, 0),      # Green
        'medium': (0, 165, 255),  # Orange  
        'low': (0, 0, 255)        # Red
    }
    
    def __init__(self, 
                 bbox_thickness=2,
                 text_scale=0.6,
                 text_thickness=2,
                 show_confidence=True,
                 show_keypoints=False):
        """
        Initialize visualizer
        
        Args:
            bbox_thickness: Thickness of bounding box
            text_scale: Scale of text
            text_thickness: Thickness of text
            show_confidence: Show confidence scores
            show_keypoints: Show pose keypoints
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        self.bbox_thickness = bbox_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.show_confidence = show_confidence
        self.show_keypoints = show_keypoints
    
    def visualize_video(self, video_path, tracking_csv, output_path):
        """
        Create visualization video with tracking results
        
        Args:
            video_path: Path to original video
            tracking_csv: Path to tracking CSV file
            output_path: Path to save output video
        """
        self.logger.info(f"Visualizing: {video_path}")
        self.logger.info(f"Tracking data: {tracking_csv}")
        
        # Load tracking data
        df = pd.read_csv(tracking_csv)
        self.logger.info(f"Loaded {len(df)} tracking records")
        
        # Group by frame
        tracking_by_frame = df.groupby('frame')
        
        # Read video
        reader = VideoReader(video_path)
        
        # Setup writer
        writer = VideoWriter(
            output_path,
            reader.fps,
            reader.width,
            reader.height
        )
        
        self.logger.info(f"Output: {output_path}")
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                
                # Get tracking data for this frame
                if frame_idx in tracking_by_frame.groups:
                    frame_data = tracking_by_frame.get_group(frame_idx)
                    
                    # Draw each person
                    for _, row in frame_data.iterrows():
                        self.draw_person(frame, row)
                    
                    # Draw summary
                    self.draw_summary(frame, frame_data)
                
                writer.write(frame)
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    self.logger.info(f"Processed {frame_idx} frames...")
        
        finally:
            reader.release()
            writer.release()
        
        self.logger.info(f"Visualization complete: {frame_idx} frames")
    
    def draw_person(self, frame, person_data):
        """
        Draw single person with bbox and info
        
        Args:
            frame: Frame to draw on
            person_data: Person data row from DataFrame
        """
        # Get data
        track_id = int(person_data['track_id'])
        x1, y1, x2, y2 = int(person_data['x1']), int(person_data['y1']), \
                         int(person_data['x2']), int(person_data['y2'])
        conf = person_data['confidence']
        score = person_data['engagement_score']
        level = person_data['engagement_level']
        
        # Get color
        color = self.COLORS[level]
        
        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Prepare label
        label_parts = [f"ID:{track_id}", f"{level.upper()}"]
        if self.show_confidence:
            label_parts.append(f"{score:.2f}")
        
        label = " | ".join(label_parts)
        
        # Draw label
        draw_text_with_background(
            frame,
            label,
            (x1, y1 - 10),
            font_scale=self.text_scale,
            thickness=self.text_thickness,
            bg_color=color,
            text_color=(255, 255, 255)
        )
    
    def draw_summary(self, frame, frame_data):
        """
        Draw engagement summary for current frame
        
        Args:
            frame: Frame to draw on
            frame_data: DataFrame with all persons in frame
        """
        # Count levels
        level_counts = frame_data['engagement_level'].value_counts().to_dict()
        
        # Total persons
        total = len(frame_data)
        
        # Prepare summary text
        summary_lines = [
            f"Students: {total}",
            f"High: {level_counts.get('high', 0)}",
            f"Med: {level_counts.get('medium', 0)}",
            f"Low: {level_counts.get('low', 0)}"
        ]
        
        # Draw summary panel
        panel_height = len(summary_lines) * 30 + 20
        panel_width = 200
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 35
        for line in summary_lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_offset += 30


def visualize_from_tracking_data(video_path, tracking_csv, output_path):
    """
    Convenience function to visualize tracking results
    
    Args:
        video_path: Path to original video
        tracking_csv: Path to tracking CSV
        output_path: Path to save output video
    """
    visualizer = TrackingVisualizer(
        bbox_thickness=2,
        text_scale=0.6,
        show_confidence=True
    )
    
    visualizer.visualize_video(video_path, tracking_csv, output_path)


def create_comparison_video(original_path, tracking_csv, output_path):
    """
    Create side-by-side comparison of original and tracked video
    
    Args:
        original_path: Path to original video
        tracking_csv: Path to tracking CSV
        output_path: Path to save comparison video
    """
    logger = setup_logger("comparison")
    logger.info("Creating comparison video...")
    
    # Load tracking data
    df = pd.read_csv(tracking_csv)
    tracking_by_frame = df.groupby('frame')
    
    # Read video
    reader = VideoReader(original_path)
    
    # Create writer (double width)
    writer = VideoWriter(
        output_path,
        reader.fps,
        reader.width * 2,  # side by side
        reader.height
    )
    
    visualizer = TrackingVisualizer()
    frame_idx = 0
    
    try:
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            
            # Create copy for visualization
            vis_frame = frame.copy()
            
            # Draw tracking on vis_frame
            if frame_idx in tracking_by_frame.groups:
                frame_data = tracking_by_frame.get_group(frame_idx)
                
                for _, row in frame_data.iterrows():
                    visualizer.draw_person(vis_frame, row)
                
                visualizer.draw_summary(vis_frame, frame_data)
            
            # Combine side by side
            combined = np.hstack([frame, vis_frame])
            
            # Add labels
            cv2.putText(
                combined, "Original",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
            cv2.putText(
                combined, "Tracked + Engagement",
                (reader.width + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
            
            writer.write(combined)
            frame_idx += 1
    
    finally:
        reader.release()
        writer.release()
    
    logger.info(f"Comparison video saved: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python visualizer.py <video_path> <tracking_csv> <output_path>")
        print("Example: python visualizer.py video.mp4 tracking.csv output.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    tracking_csv = sys.argv[2]
    output_path = sys.argv[3]
    
    # Create standard visualization
    visualize_from_tracking_data(video_path, tracking_csv, output_path)
    
    # Optionally create comparison
    comparison_path = output_path.replace('.mp4', '_comparison.mp4')
    create_comparison_video(video_path, tracking_csv, comparison_path)