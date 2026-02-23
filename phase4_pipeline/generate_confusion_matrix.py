"""
Generate Confusion Matrix for Engagement Classification
Supports low, high, and medium engagement levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


class ConfusionMatrixGenerator:
    """Generate confusion matrix for engagement predictions"""
    
    # Class mappings
    CLASS_ORDER = ['high', 'medium', 'low']
    CLASS_LABELS = ['High\n(Engaged)', 'Medium\n(Moderately)', 'Low\n(Disengaged)']
    
    def __init__(self, predictions_csv, ground_truth_csv=None, output_dir='confusion_matrix_analysis'):
        """
        Initialize confusion matrix generator
        
        Args:
            predictions_csv: CSV file with predictions (must have 'engagement_level' column)
            ground_truth_csv: CSV file with ground truth labels (optional)
            output_dir: Directory to save outputs
        """
        self.predictions_csv = Path(predictions_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predictions
        print(f"📂 Loading predictions from {self.predictions_csv}...")
        self.pred_df = pd.read_csv(predictions_csv)
        
        # Clean column names (remove leading/trailing spaces)
        self.pred_df.columns = self.pred_df.columns.str.strip()
        
        # Verify columns
        if 'engagement_level' not in self.pred_df.columns:
            raise ValueError("Predictions CSV must have 'engagement_level' column")
        
        # Load ground truth if provided
        self.has_ground_truth = False
        if ground_truth_csv and Path(ground_truth_csv).exists():
            print(f"📂 Loading ground truth from {ground_truth_csv}...")
            self.gt_df = pd.read_csv(ground_truth_csv)
            
            # Clean column names (remove leading/trailing spaces)
            self.gt_df.columns = self.gt_df.columns.str.strip()
            
            self.has_ground_truth = True
            
            # Merge on common keys
            self._merge_data()
        else:
            print("⚠️  No ground truth provided - will only show prediction distribution")
            self.merged_df = self.pred_df
        
        print(f"✓ Loaded {len(self.merged_df)} samples")
    
    def _merge_data(self):
        """Merge predictions with ground truth"""
        # Determine merge keys based on available columns
        merge_keys = []
        
        if 'frame' in self.pred_df.columns and 'frame' in self.gt_df.columns:
            merge_keys.append('frame')
        
        if 'track_id' in self.pred_df.columns and 'track_id' in self.gt_df.columns:
            merge_keys.append('track_id')
        
        if not merge_keys:
            raise ValueError("Cannot find common merge keys (frame, track_id) in both CSVs")
        
        print(f"🔗 Merging on keys: {merge_keys}")
        
        # Rename ground truth column to avoid conflicts
        gt_columns = self.gt_df.columns.tolist()
        if 'engagement_level' in gt_columns:
            self.gt_df = self.gt_df.rename(columns={'engagement_level': 'ground_truth'})
        elif 'label' in gt_columns:
            self.gt_df = self.gt_df.rename(columns={'label': 'ground_truth'})
        elif 'true_label' in gt_columns:
            self.gt_df = self.gt_df.rename(columns={'true_label': 'ground_truth'})
        elif 'ground_truth' in gt_columns:
            # Already has ground_truth column, keep it
            pass
        else:
            raise ValueError(f"Ground truth CSV must have one of: 'engagement_level', 'label', or 'true_label'. Found columns: {gt_columns}")
        
        # Merge
        self.merged_df = pd.merge(
            self.pred_df,
            self.gt_df[merge_keys + ['ground_truth']],
            on=merge_keys,
            how='inner'
        )
        
        print(f"✓ Merged {len(self.merged_df)} matching samples")
        
        if len(self.merged_df) == 0:
            raise ValueError("No matching samples found! Check your merge keys.")
    
    def plot_confusion_matrix(self, normalize=False, save_name='confusion_matrix.png'):
        """
        Generate and plot confusion matrix
        
        Args:
            normalize: Whether to normalize the confusion matrix
            save_name: Filename for the output plot
        """
        if not self.has_ground_truth:
            print("❌ Cannot generate confusion matrix without ground truth data")
            return None
        
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRIX")
        print("="*80)
        
        # Get true and predicted labels
        y_true = self.merged_df['ground_truth'].values
        y_pred = self.merged_df['engagement_level'].values
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.CLASS_ORDER)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.CLASS_LABELS,
            yticklabels=self.CLASS_LABELS,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'},
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {output_path}")
        plt.close()
        
        return cm
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if not self.has_ground_truth:
            print("❌ Cannot generate classification report without ground truth data")
            return None
        
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        
        y_true = self.merged_df['ground_truth'].values
        y_pred = self.merged_df['engagement_level'].values
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed report
        print("\n" + "-"*80)
        report = classification_report(
            y_true,
            y_pred,
            labels=self.CLASS_ORDER,
            target_names=self.CLASS_ORDER,
            digits=4
        )
        print(report)
        
        # Save report
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write(report)
        print(f"✓ Saved classification report to {report_path}")
        
        # Generate per-class metrics for CSV
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.CLASS_ORDER, average=None
        )
        
        metrics_df = pd.DataFrame({
            'class': self.CLASS_ORDER,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        
        metrics_path = self.output_dir / 'per_class_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Saved per-class metrics to {metrics_path}")
        
        return report
    
    def plot_prediction_distribution(self):
        """Plot distribution of predictions"""
        print("\n📊 Generating prediction distribution plot...")
        
        fig, axes = plt.subplots(1, 2 if self.has_ground_truth else 1, figsize=(14 if self.has_ground_truth else 8, 5))
        
        if not self.has_ground_truth:
            axes = [axes]
        
        # Prediction distribution
        pred_dist = self.merged_df['engagement_level'].value_counts()
        pred_dist = pred_dist.reindex(self.CLASS_ORDER, fill_value=0)
        
        colors = {
            'high': '#2ecc71',
            'medium': '#f39c12',
            'low': '#e74c3c'
        }
        bar_colors = [colors.get(c, '#95a5a6') for c in pred_dist.index]
        
        axes[0].bar(range(len(pred_dist)), pred_dist.values, color=bar_colors)
        axes[0].set_xticks(range(len(pred_dist)))
        axes[0].set_xticklabels(self.CLASS_LABELS, fontsize=10)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for i, v in enumerate(pred_dist.values):
            axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Ground truth distribution if available
        if self.has_ground_truth:
            gt_dist = self.merged_df['ground_truth'].value_counts()
            gt_dist = gt_dist.reindex(self.CLASS_ORDER, fill_value=0)
            
            axes[1].bar(range(len(gt_dist)), gt_dist.values, color=bar_colors)
            axes[1].set_xticks(range(len(gt_dist)))
            axes[1].set_xticklabels(self.CLASS_LABELS, fontsize=10)
            axes[1].set_ylabel('Count', fontsize=12)
            axes[1].set_title('Ground Truth Distribution', fontsize=14, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            
            # Add counts on bars
            for i, v in enumerate(gt_dist.values):
                axes[1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'distribution_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved distribution plot to {plot_path}")
        plt.close()
    
    def plot_per_class_accuracy(self):
        """Plot accuracy per class (bar chart)"""
        if not self.has_ground_truth:
            return
        
        print("\n📊 Generating per-class accuracy plot...")
        
        y_true = self.merged_df['ground_truth'].values
        y_pred = self.merged_df['engagement_level'].values
        
        # Calculate per-class accuracy
        accuracies = []
        for cls in self.CLASS_ORDER:
            mask = y_true == cls
            if mask.sum() > 0:
                cls_accuracy = (y_pred[mask] == y_true[mask]).mean()
                accuracies.append(cls_accuracy * 100)
            else:
                accuracies.append(0)
        
        # Plot
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(self.CLASS_ORDER)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
        plt.xticks(range(len(self.CLASS_ORDER)), self.CLASS_LABELS, fontsize=11)
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, acc + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'per_class_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class accuracy to {plot_path}")
        plt.close()
    
    def generate_complete_analysis(self):
        """Generate all confusion matrix and analysis plots"""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)
        print(f"Predictions from: {self.predictions_csv}")
        if self.has_ground_truth:
            print(f"Ground truth available: ✓")
        else:
            print(f"Ground truth available: ✗ (only distribution analysis)")
        print("="*80)
        
        # Generate all analyses
        self.plot_prediction_distribution()
        
        if self.has_ground_truth:
            # Both normalized and non-normalized confusion matrices
            self.plot_confusion_matrix(normalize=False, save_name='confusion_matrix.png')
            self.plot_confusion_matrix(normalize=True, save_name='confusion_matrix_normalized.png')
            
            # Classification report
            self.generate_classification_report()
            
            # Per-class accuracy
            self.plot_per_class_accuracy()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print(f"✓ All outputs saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix for engagement classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With ground truth
  python generate_confusion_matrix.py --predictions sav2.csv --ground_truth ground_truth.csv
  
  # Without ground truth (only distribution)
  python generate_confusion_matrix.py --predictions sav2.csv
  
  # Custom output directory
  python generate_confusion_matrix.py --predictions sav2.csv --ground_truth gt.csv --output my_analysis
        """
    )
    
    parser.add_argument('--predictions', type=str, required=True,
                       help='CSV file with predictions (must have "engagement_level" column)')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='CSV file with ground truth labels (optional)')
    parser.add_argument('--output', type=str, default='confusion_matrix_analysis',
                       help='Output directory for results (default: confusion_matrix_analysis)')
    
    args = parser.parse_args()
    
    # Generate confusion matrix
    generator = ConfusionMatrixGenerator(
        predictions_csv=args.predictions,
        ground_truth_csv=args.ground_truth,
        output_dir=args.output
    )
    
    generator.generate_complete_analysis()


if __name__ == "__main__":
    main()
