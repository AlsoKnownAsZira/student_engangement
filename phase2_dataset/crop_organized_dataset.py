"""
Crop Persons with Split Preservation
Crops persons from organized dataset while maintaining train/val/test splits
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import shutil

import config
from utils.logger import setup_logger, ProgressLogger


def crop_persons_from_frame(frame, detector, min_width=50, min_height=100):
    """
    Detect and crop all persons from a frame
    
    Args:
        frame: Input frame
        detector: YOLO detector
        min_width: Minimum person width
        min_height: Minimum person height
    
    Returns:
        List of cropped person images with metadata
    """
    # Detect persons
    results = detector(frame, classes=[0], verbose=False)  # class 0 = person
    
    crops = []
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            
            # Check minimum size
            width = x2 - x1
            height = y2 - y1
            
            if width >= min_width and height >= min_height:
                # Crop person
                person_crop = frame[y1:y2, x1:x2]
                
                crops.append({
                    'crop': person_crop,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'width': width,
                    'height': height
                })
    
    return crops


def process_split_category(input_dir, output_dir, detector, 
                           split_name, category_name,
                           target_size=224, min_width=50, min_height=100):
    """
    Process one split-category combination
    
    Args:
        input_dir: Input directory (e.g., train/High)
        output_dir: Output directory (e.g., person_crops/train/high)
        detector: YOLO detector
        split_name: 'train', 'val', or 'test'
        category_name: 'High', 'Medium', or 'Low'
        target_size: Target size for crops
        min_width: Minimum person width
        min_height: Minimum person height
    
    Returns:
        Statistics
    """
    logger = setup_logger(f"crop_{split_name}_{category_name}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(input_dir.glob(f'*{ext}'))
    
    logger.info(f"\n{split_name.upper()} / {category_name.upper()}")
    logger.info(f"  Found {len(image_files)} frames")
    
    total_persons = 0
    frames_with_persons = 0
    
    progress = ProgressLogger(len(image_files), desc=f"{split_name}/{category_name}", logger=logger)
    
    for frame_path in image_files:
        # Read frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Crop persons
        crops = crop_persons_from_frame(
            frame, 
            detector,
            min_width=min_width,
            min_height=min_height
        )
        
        if crops:
            frames_with_persons += 1
        
        # Save crops
        frame_name = frame_path.stem
        for i, crop_data in enumerate(crops):
            person_crop = crop_data['crop']
            
            # Resize maintaining aspect ratio
            h, w = person_crop.shape[:2]
            if h > w:
                new_h = target_size
                new_w = int(w * target_size / h)
            else:
                new_w = target_size
                new_h = int(h * target_size / w)
            
            resized = cv2.resize(person_crop, (new_w, new_h))
            
            # Create square image with padding
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Save with descriptive name
            output_path = output_dir / f"{frame_name}_p{i}.jpg"
            cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            total_persons += 1
        
        progress.update()
    
    progress.finish()
    
    logger.info(f"  Extracted {total_persons} person crops from {frames_with_persons} frames")
    
    return {
        'split': split_name,
        'category': category_name,
        'total_frames': len(image_files),
        'frames_with_persons': frames_with_persons,
        'total_persons': total_persons
    }


def crop_organized_dataset(dataset_root, output_root, detector,
                          target_size=224, min_width=50, min_height=100):
    """
    Crop persons from organized dataset (train/val/test × categories)
    
    Args:
        dataset_root: Root directory with train/val/test folders
        output_root: Output root directory
        detector: YOLO detector
        target_size: Target size for crops
        min_width: Minimum person width
        min_height: Minimum person height
    
    Returns:
        Overall statistics
    """
    logger = setup_logger("crop_organized")
    
    logger.info("=" * 80)
    logger.info("CROP PERSONS - ORGANIZED DATASET")
    logger.info("=" * 80)
    logger.info(f"Input: {dataset_root}")
    logger.info(f"Output: {output_root}")
    logger.info("=" * 80)
    
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # Define splits and categories
    splits = ['train', 'val', 'test']
    categories = ['High', 'Medium', 'Low']
    
    all_stats = []
    total_frames = 0
    total_persons = 0
    
    # Process each split-category combination
    for split in splits:
        for category in categories:
            input_dir = dataset_root / split / category
            output_dir = output_root / split / category.lower()
            
            if not input_dir.exists():
                logger.warning(f"Directory not found: {input_dir}")
                continue
            
            stats = process_split_category(
                input_dir=input_dir,
                output_dir=output_dir,
                detector=detector,
                split_name=split,
                category_name=category,
                target_size=target_size,
                min_width=min_width,
                min_height=min_height
            )
            
            all_stats.append(stats)
            total_frames += stats['total_frames']
            total_persons += stats['total_persons']
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CROPPING SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal frames processed: {total_frames:,}")
    logger.info(f"Total person crops: {total_persons:,}")
    
    logger.info(f"\nBreakdown by split:")
    for split in splits:
        split_stats = [s for s in all_stats if s['split'] == split]
        split_persons = sum(s['total_persons'] for s in split_stats)
        logger.info(f"  {split.upper()}: {split_persons:,} person crops")
        
        for stat in split_stats:
            logger.info(f"    {stat['category']}: {stat['total_persons']:,}")
    
    logger.info(f"\nOutput structure:")
    logger.info(f"  {output_root}/")
    for split in splits:
        logger.info(f"    {split}/")
        for category in ['high', 'medium', 'low']:
            output_dir = output_root / split / category
            if output_dir.exists():
                count = len(list(output_dir.glob('*.jpg')))
                logger.info(f"      {category}/ ({count:,} crops)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ READY FOR TRAINING!")
    logger.info("=" * 80)
    logger.info(f"\nDataset location: {output_root}")
    logger.info(f"Next step: cd ../phase3_training")
    logger.info(f"           python train_classifier.py --data {output_root}")
    logger.info("=" * 80)
    
    return {
        'total_frames': total_frames,
        'total_persons': total_persons,
        'stats': all_stats
    }


def create_data_yaml(output_root):
    """Create data.yaml for YOLO training"""
    output_root = Path(output_root)
    yaml_path = output_root / 'data.yaml'
    
    yaml_content = f"""# Person Engagement Classification Dataset
# Cropped from organized frame dataset

path: {output_root}
train: train
val: val
test: test

# Classes (lowercase to match folders)
names:
  0: high
  1: low
  2: medium
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Crop persons from organized dataset (train/val/test)'
    )
    
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory with train/val/test folders'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for person crops'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11s.pt',
        help='Detection model (default: yolo11s.pt)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=224,
        help='Target size for crops (default: 224)'
    )
    
    parser.add_argument(
        '--min-width',
        type=int,
        default=50,
        help='Minimum person width (default: 50)'
    )
    
    parser.add_argument(
        '--min-height',
        type=int,
        default=100,
        help='Minimum person height (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Load detector
    logger = setup_logger("main")
    logger.info(f"Loading detector: {args.model}")
    detector = YOLO(args.model)
    
    # Process dataset
    stats = crop_organized_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output,
        detector=detector,
        target_size=args.target_size,
        min_width=args.min_width,
        min_height=args.min_height
    )
    
    # Create data.yaml
    yaml_path = create_data_yaml(args.output)
    logger.info(f"\n✅ Created data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()