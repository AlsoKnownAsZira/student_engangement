"""
Phase 3: Evaluate Classifier
Evaluate trained engagement classifier on test set
"""

import sys
sys.path.append('..')

import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from utils.logger import setup_logger


def evaluate_classifier(model_path, data_path, output_dir=None, split='test'):
    """
    Evaluate classifier on dataset
    
    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        output_dir: Output directory for results
        split: Which split to evaluate ('test', 'val', or 'train')
    
    Returns:
        Evaluation metrics
    """
    logger = setup_logger("evaluation")
    
    logger.info("=" * 80)
    logger.info("PHASE 3: Model Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Split: {split}")
    
    # Load model
    model = YOLO(model_path)
    
    # Validate/test
    logger.info("\nRunning evaluation...\n")
    
    results = model.val(
        data=str(data_path),
        split=split,
        save_json=True,
        plots=True
    )
    
    # Extract metrics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Top-1 and Top-5 accuracy
    top1 = results.top1
    top5 = results.top5
    
    logger.info(f"\nAccuracy:")
    logger.info(f"  Top-1: {top1:.4f}")
    logger.info(f"  Top-5: {top5:.4f}")
    
    # Per-class metrics (if available)
    if hasattr(results, 'confusion_matrix'):
        cm = results.confusion_matrix.matrix
        logger.info(f"\nConfusion Matrix:")
        logger.info(cm)
    
    # Save detailed report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_dict = {
            'top1_accuracy': float(top1),
            'top5_accuracy': float(top5),
        }
        
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_path = output_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"\nSaved metrics: {metrics_path}")
        
        # Plot confusion matrix if available
        if hasattr(results, 'confusion_matrix'):
            plot_confusion_matrix(
                results.confusion_matrix.matrix,
                ['high', 'low', 'medium'],
                output_dir / 'confusion_matrix.png'
            )
            logger.info(f"Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")
    
    logger.info("=" * 80)
    
    return {
        'top1': float(top1),
        'top5': float(top5),
        'results': results
    }


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Normalized Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(model_paths, data_path, output_dir=None):
    """
    Compare multiple models
    
    Args:
        model_paths: List of model paths
        data_path: Path to dataset
        output_dir: Output directory for comparison
    
    Returns:
        Comparison results
    """
    logger = setup_logger("model_comparison")
    
    logger.info("=" * 80)
    logger.info(f"MODEL COMPARISON ({len(model_paths)} models)")
    logger.info("=" * 80)
    
    results = []
    
    for i, model_path in enumerate(model_paths, 1):
        logger.info(f"\n--- Model {i}/{len(model_paths)}: {Path(model_path).name} ---")
        
        try:
            metrics = evaluate_classifier(
                model_path,
                data_path,
                output_dir=None,
                split='test'
            )
            
            results.append({
                'model': Path(model_path).name,
                'model_path': str(model_path),
                'top1': metrics['top1'],
                'top5': metrics['top5']
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {model_path}: {e}")
    
    # Create comparison table
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('top1', ascending=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 80)
        logger.info("\n" + df.to_string(index=False))
        
        # Save
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_path = output_dir / 'model_comparison.csv'
            df.to_csv(comparison_path, index=False)
            logger.info(f"\nSaved comparison: {comparison_path}")
        
        logger.info("=" * 80)
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 3: Evaluate engagement classifier'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (best.pt or last.pt)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset directory or data.yaml'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate (default: test)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple models (provide multiple model paths)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output
    if output_dir is None and not args.compare:
        # Create output dir based on model path
        model_dir = Path(args.model).parent.parent
        output_dir = model_dir / 'evaluation'
    
    if args.compare:
        # Compare multiple models
        compare_models(args.compare, args.data, output_dir)
    else:
        # Single model evaluation
        evaluate_classifier(
            args.model,
            args.data,
            output_dir=output_dir,
            split=args.split
        )


if __name__ == "__main__":
    main()