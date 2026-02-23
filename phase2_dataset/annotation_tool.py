"""
Phase 2: Simple Annotation Tool
Quick tool to label person crops with engagement levels
"""

import sys
sys.path.append('..')

import cv2
import os
from pathlib import Path
import argparse
import json
from collections import defaultdict

import config
from utils.logger import setup_logger


class SimpleAnnotator:
    """Simple keyboard-based annotation tool"""
    
    def __init__(self, images_dir, output_dir):
        """
        Initialize annotator
        
        Args:
            images_dir: Directory containing person crop images
            output_dir: Output directory for organized images
        """
        self.logger = setup_logger("annotator")
        
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        for level in ['high', 'medium', 'low', 'skip']:
            (self.output_dir / level).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.images.extend(self.images_dir.glob(f'**/*{ext}'))
        
        self.images = sorted(self.images)
        self.current_idx = 0
        
        # Load progress if exists
        self.progress_file = self.output_dir / 'annotation_progress.json'
        self.annotations = self.load_progress()
        
        self.logger.info(f"Found {len(self.images)} images")
        self.logger.info(f"Already annotated: {len(self.annotations)}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def load_progress(self):
        """Load annotation progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save annotation progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def copy_image(self, image_path, label):
        """Copy image to labeled directory"""
        dest_dir = self.output_dir / label
        dest_path = dest_dir / image_path.name
        
        # If file exists, add number suffix
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        # Copy file
        import shutil
        shutil.copy2(image_path, dest_path)
        
        return str(dest_path)
    
    def annotate(self):
        """Main annotation loop"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANNOTATION TOOL")
        self.logger.info("=" * 80)
        self.logger.info("Controls:")
        self.logger.info("  H = High engagement")
        self.logger.info("  M = Medium engagement")
        self.logger.info("  L = Low engagement")
        self.logger.info("  S = Skip (unclear/bad crop)")
        self.logger.info("  B = Go back to previous image")
        self.logger.info("  Q = Quit and save progress")
        self.logger.info("=" * 80)
        
        # Skip already annotated
        while self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            if str(image_path) not in self.annotations:
                break
            self.current_idx += 1
        
        window_name = "Annotation Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        
        while self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.warning(f"Cannot read: {image_path}")
                self.current_idx += 1
                continue
            
            # Create display image (larger for visibility)
            display_img = cv2.resize(img, (600, 600))
            
            # Add info text
            info_text = [
                f"Image {self.current_idx + 1}/{len(self.images)}",
                f"File: {image_path.name}",
                "",
                "H = High | M = Medium | L = Low",
                "S = Skip | B = Back | Q = Quit"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(
                    display_img, text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )
                y_offset += 30
            
            cv2.imshow(window_name, display_img)
            
            # Get key
            key = cv2.waitKey(0) & 0xFF
            
            # Process key
            label = None
            if key == ord('h') or key == ord('H'):
                label = 'high'
            elif key == ord('m') or key == ord('M'):
                label = 'medium'
            elif key == ord('l') or key == ord('L'):
                label = 'low'
            elif key == ord('s') or key == ord('S'):
                label = 'skip'
            elif key == ord('b') or key == ord('B'):
                # Go back
                if self.current_idx > 0:
                    self.current_idx -= 1
                    # Remove previous annotation
                    prev_path = str(self.images[self.current_idx])
                    if prev_path in self.annotations:
                        del self.annotations[prev_path]
                continue
            elif key == ord('q') or key == ord('Q'):
                # Quit
                break
            
            if label:
                # Copy image to labeled directory
                dest_path = self.copy_image(image_path, label)
                
                # Save annotation
                self.annotations[str(image_path)] = {
                    'label': label,
                    'dest_path': dest_path
                }
                
                self.logger.info(f"[{self.current_idx + 1}/{len(self.images)}] "
                               f"{image_path.name} -> {label.upper()}")
                
                # Save progress periodically
                if len(self.annotations) % 10 == 0:
                    self.save_progress()
                
                self.current_idx += 1
        
        cv2.destroyAllWindows()
        
        # Final save
        self.save_progress()
        
        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ANNOTATION SUMMARY")
        self.logger.info("=" * 80)
        
        label_counts = defaultdict(int)
        for data in self.annotations.values():
            label_counts[data['label']] += 1
        
        self.logger.info(f"Total annotated: {len(self.annotations)}")
        for label, count in sorted(label_counts.items()):
            self.logger.info(f"  {label.upper()}: {count}")
        
        remaining = len(self.images) - len(self.annotations)
        self.logger.info(f"Remaining: {remaining}")
        
        return self.annotations


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 2: Simple annotation tool for person crops'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing person crop images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: outputs/annotations)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output or config.ANNOTATIONS_DIR
    
    # Run annotator
    annotator = SimpleAnnotator(args.images, output_dir)
    annotator.annotate()
    
    print("\nAnnotation complete!")
    print(f"Labeled images saved to: {output_dir}")
    print(f"Progress saved to: {annotator.progress_file}")


if __name__ == "__main__":
    main()