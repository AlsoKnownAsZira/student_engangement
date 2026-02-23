"""
Phase 2: Prepare Dataset
Organize annotated data into train/val/test splits
"""

import sys
sys.path.append('..')

import os
import shutil
from pathlib import Path
import argparse
import random
from collections import defaultdict

import config
from utils.logger import setup_logger


def count_images_in_dir(directory):
    """Count image files in directory"""
    directory = Path(directory)
    count = 0
    for ext in ['.jpg', '.jpeg', '.png']:
        count += len(list(directory.glob(f'*{ext}')))
    return count


def split_dataset(annotations_dir, output_dir, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42):
    """
    Split annotated dataset into train/val/test
    
    Args:
        annotations_dir: Directory with labeled subdirectories (high/medium/low)
        output_dir: Output directory for organized dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    
    Returns:
        Statistics dictionary
    """
    logger = setup_logger("dataset_prep")
    
    logger.info("=" * 80)
    logger.info("PHASE 2: Dataset Preparation")
    logger.info("=" * 80)
    
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    
    # Check ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    logger.info(f"Input: {annotations_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Split: Train {train_ratio:.0%}, Val {val_ratio:.0%}, Test {test_ratio:.0%}")
    
    random.seed(seed)
    
    # Classes to process (skip 'skip' folder)
    classes = ['high', 'medium', 'low']
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_name in classes:
        class_dir = annotations_dir / class_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        # Get all images
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(class_dir.glob(f'*{ext}'))
        
        if not images:
            logger.warning(f"No images found in {class_dir}")
            continue
        
        logger.info(f"\nClass: {class_name.upper()}")
        logger.info(f"  Total images: {len(images)}")
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        logger.info(f"  Train: {len(train_images)}")
        logger.info(f"  Val: {len(val_images)}")
        logger.info(f"  Test: {len(test_images)}")
        
        # Copy files to output directories
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            if not split_images:
                continue
            
            # Create output directory
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_path in split_images:
                dest_path = split_dir / img_path.name
                
                # Handle duplicates
                if dest_path.exists():
                    stem = dest_path.stem
                    suffix = dest_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = split_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                shutil.copy2(img_path, dest_path)
            
            stats[split_name][class_name] = len(split_images)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)
    
    for split_name in ['train', 'val', 'test']:
        total = sum(stats[split_name].values())
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Total: {total}")
        for class_name in classes:
            count = stats[split_name][class_name]
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"    {class_name}: {count} ({pct:.1f}%)")
    
    # Grand total
    grand_total = sum(
        sum(stats[split].values()) 
        for split in stats
    )
    logger.info(f"\nGrand Total: {grand_total} images")
    
    # Create data.yaml for YOLO training
    yaml_path = output_dir / 'data.yaml'
    create_data_yaml(output_dir, yaml_path, classes)
    logger.info(f"\nCreated data.yaml: {yaml_path}")
    
    logger.info("=" * 80)
    
    return dict(stats)


def create_data_yaml(dataset_dir, output_path, classes):
    """Create data.yaml for YOLO classification training"""
    yaml_content = f"""# Person Engagement Classification Dataset
# Generated by Phase 2 Dataset Preparation

path: {dataset_dir}  # dataset root dir
train: train  # train images (relative to 'path')
val: val  # val images (relative to 'path')
test: test  # test images (relative to 'path')

# Classes
names:
  0: high
  1: low
  2: medium
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)


def verify_dataset(dataset_dir):
    """Verify dataset structure and counts"""
    logger = setup_logger("verify")
    
    logger.info("\n" + "=" * 80)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 80)
    
    dataset_dir = Path(dataset_dir)
    
    splits = ['train', 'val', 'test']
    classes = ['high', 'medium', 'low']
    
    all_good = True
    
    for split in splits:
        split_dir = dataset_dir / split
        
        if not split_dir.exists():
            logger.error(f"Missing split directory: {split_dir}")
            all_good = False
            continue
        
        logger.info(f"\n{split.upper()}:")
        
        for class_name in classes:
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"  Missing class directory: {class_name}")
                continue
            
            count = count_images_in_dir(class_dir)
            logger.info(f"  {class_name}: {count} images")
            
            if count == 0:
                logger.warning(f"  No images in {class_name}!")
    
    # Check data.yaml
    yaml_path = dataset_dir / 'data.yaml'
    if yaml_path.exists():
        logger.info(f"\n✓ data.yaml exists")
    else:
        logger.warning(f"\n✗ data.yaml not found")
        all_good = False
    
    if all_good:
        logger.info("\n✓ Dataset verification passed!")
    else:
        logger.warning("\n⚠ Dataset verification found issues")
    
    logger.info("=" * 80)
    
    return all_good


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 2: Prepare and organize annotated dataset'
    )
    
    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Directory with annotated images (high/medium/low subdirs)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized dataset'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=config.TRAIN_RATIO,
        help=f'Training set ratio (default: {config.TRAIN_RATIO})'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=config.VAL_RATIO,
        help=f'Validation set ratio (default: {config.VAL_RATIO})'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=config.TEST_RATIO,
        help=f'Test set ratio (default: {config.TEST_RATIO})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset after creation'
    )
    
    args = parser.parse_args()
    
    # Split dataset
    split_dataset(
        args.annotations,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Verify if requested
    if args.verify:
        verify_dataset(args.output)


if __name__ == "__main__":
    main()