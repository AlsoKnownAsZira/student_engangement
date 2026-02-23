"""
Phase 3: Train Engagement Classifier
Train YOLOv11 classifier on per-person engagement dataset
"""

import sys
sys.path.append('..')

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

import config
from utils.logger import setup_logger, ExperimentLogger


def train_classifier(data_path, output_dir, model_name='yolo11s-cls.pt', **train_args):
    """
    Train engagement classifier
    
    Args:
        data_path: Path to dataset (directory with train/val/test or data.yaml)
        output_dir: Output directory for training results
        model_name: Base model name
        **train_args: Additional training arguments
    
    Returns:
        Training results
    """
    logger = setup_logger("training")
    
    logger.info("=" * 80)
    logger.info("PHASE 3: Training Engagement Classifier")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading base model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare training arguments
    default_args = config.TRAINING_CONFIG.copy()
    default_args.update(train_args)
    default_args['data'] = str(data_path)
    default_args['project'] = str(output_dir)
    
    logger.info("\nTraining Configuration:")
    for key, value in default_args.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training...")
    logger.info("=" * 80 + "\n")
    
    # Train
    results = model.train(**default_args)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    
    # Get best model path
    run_dir = Path(default_args['project']) / default_args.get('name', 'train')
    best_model = run_dir / 'weights' / 'best.pt'
    last_model = run_dir / 'weights' / 'last.pt'
    
    logger.info(f"\nBest model: {best_model}")
    logger.info(f"Last model: {last_model}")
    logger.info(f"Results directory: {run_dir}")
    
    return {
        'results': results,
        'best_model': str(best_model),
        'last_model': str(last_model),
        'run_dir': str(run_dir)
    }


def resume_training(checkpoint_path, **train_args):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint (last.pt)
        **train_args: Additional training arguments
    
    Returns:
        Training results
    """
    logger = setup_logger("resume_training")
    
    logger.info("=" * 80)
    logger.info("PHASE 3: Resuming Training")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    model = YOLO(checkpoint_path)
    
    # Resume training
    logger.info("\nResuming training...\n")
    results = model.train(resume=True, **train_args)
    
    logger.info("\nTraining resumed and complete!")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 3: Train person engagement classifier'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset directory or data.yaml file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=config.TRAINING_CONFIG['model'],
        help=f"Base model (default: {config.TRAINING_CONFIG['model']})"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.TRAINING_CONFIG['epochs'],
        help=f"Number of epochs (default: {config.TRAINING_CONFIG['epochs']})"
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=config.TRAINING_CONFIG['batch'],
        help=f"Batch size (default: {config.TRAINING_CONFIG['batch']})"
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=config.TRAINING_CONFIG['imgsz'],
        help=f"Image size (default: {config.TRAINING_CONFIG['imgsz']})"
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=config.TRAINING_CONFIG['patience'],
        help=f"Early stopping patience (default: {config.TRAINING_CONFIG['patience']})"
    )
    
    parser.add_argument(
        '--device',
        default=config.TRAINING_CONFIG['device'],
        help=f"Device (default: {config.TRAINING_CONFIG['device']})"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: outputs/trained_models)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='engagement_classifier',
        help='Experiment name (default: engagement_classifier)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint (path to last.pt)'
    )
    
    # Additional YOLO training arguments
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='Save checkpoint every N epochs (-1 = disabled)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for data loading (default: 8)'
    )
    
    parser.add_argument(
        '--cache',
        type=str,
        default=None,
        choices=['ram', 'disk', None],
        help='Cache images in RAM or disk for faster training'
    )
    
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Use Automatic Mixed Precision (AMP) for faster training'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        help='Optimizer (auto, SGD, Adam, AdamW, etc.)'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate (default: 0.01)'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Allow overwriting existing experiment'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output or config.TRAINED_MODELS_DIR
    
    if args.resume:
        # Resume training
        resume_training(args.resume)
    else:
        # New training
        train_args = {
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'patience': args.patience,
            'device': args.device,
            'name': args.name,
            'workers': args.workers,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'exist_ok': args.exist_ok,
            'amp': args.amp
        }
        
        # Add optional arguments if specified
        if args.save_period > 0:
            train_args['save_period'] = args.save_period
        
        if args.cache:
            train_args['cache'] = args.cache
        
        train_classifier(
            args.data,
            output_dir,
            model_name=args.model,
            **train_args
        )


if __name__ == "__main__":
    main()