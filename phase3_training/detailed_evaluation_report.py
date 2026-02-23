import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', 
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class DetailedEvaluationReport:
    def __init__(self, confusion_matrix, class_names, output_dir):
        """
        Initialize detailed evaluation report generator
        
        Args:
            confusion_matrix: numpy array of confusion matrix
            class_names: list of class names
            output_dir: directory to save reports
        """
        self.cm = confusion_matrix
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        self.calculate_metrics()
        
    def calculate_metrics(self):
        """Calculate detailed metrics for each class"""
        # Handle classes with no samples
        valid_classes = self.cm.sum(axis=1) > 0
        
        # Per-class metrics
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.support = []
        
        for i in range(len(self.class_names)):
            if not valid_classes[i]:
                self.precision.append(0.0)
                self.recall.append(0.0)
                self.f1_score.append(0.0)
                self.support.append(0)
                continue
                
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp
            
            # Precision
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            self.precision.append(prec)
            
            # Recall
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            self.recall.append(rec)
            
            # F1-Score
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            self.f1_score.append(f1)
            
            # Support
            self.support.append(self.cm[i, :].sum())
        
        # Overall metrics
        self.total_samples = self.cm.sum()
        self.correct_predictions = np.trace(self.cm)
        self.accuracy = self.correct_predictions / self.total_samples if self.total_samples > 0 else 0
        
        # Macro averages
        valid_metrics = [(p, r, f) for p, r, f, s in zip(self.precision, self.recall, self.f1_score, self.support) if s > 0]
        if valid_metrics:
            self.macro_precision = np.mean([m[0] for m in valid_metrics])
            self.macro_recall = np.mean([m[1] for m in valid_metrics])
            self.macro_f1 = np.mean([m[2] for m in valid_metrics])
        else:
            self.macro_precision = 0.0
            self.macro_recall = 0.0
            self.macro_f1 = 0.0
        
        # Weighted averages
        total_support = sum(self.support)
        if total_support > 0:
            self.weighted_precision = sum(p * s for p, s in zip(self.precision, self.support)) / total_support
            self.weighted_recall = sum(r * s for r, s in zip(self.recall, self.support)) / total_support
            self.weighted_f1 = sum(f * s for f, s in zip(self.f1_score, self.support)) / total_support
        else:
            self.weighted_precision = 0.0
            self.weighted_recall = 0.0
            self.weighted_f1 = 0.0
    
    def print_detailed_report(self):
        """Print detailed evaluation report to console"""
        logger.info("\n" + "="*100)
        logger.info("DETAILED EVALUATION REPORT")
        logger.info("="*100)
        
        # Overall metrics
        logger.info("\n📊 OVERALL PERFORMANCE")
        logger.info("-" * 100)
        logger.info(f"Total Samples: {self.total_samples:,}")
        logger.info(f"Correct Predictions: {self.correct_predictions:,}")
        logger.info(f"Incorrect Predictions: {self.total_samples - self.correct_predictions:,}")
        logger.info(f"Overall Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        logger.info("")
        logger.info(f"Macro-averaged Precision: {self.macro_precision:.4f}")
        logger.info(f"Macro-averaged Recall: {self.macro_recall:.4f}")
        logger.info(f"Macro-averaged F1-Score: {self.macro_f1:.4f}")
        logger.info("")
        logger.info(f"Weighted-averaged Precision: {self.weighted_precision:.4f}")
        logger.info(f"Weighted-averaged Recall: {self.weighted_recall:.4f}")
        logger.info(f"Weighted-averaged F1-Score: {self.weighted_f1:.4f}")
        
        # Per-class metrics
        logger.info("\n📋 PER-CLASS PERFORMANCE")
        logger.info("-" * 100)
        logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12} {'Accuracy':<12}")
        logger.info("-" * 100)
        
        for i, class_name in enumerate(self.class_names):
            if self.support[i] > 0:
                class_acc = self.cm[i, i] / self.support[i]
                logger.info(f"{class_name:<20} {self.precision[i]:<12.4f} {self.recall[i]:<12.4f} "
                          f"{self.f1_score[i]:<12.4f} {self.support[i]:<12,} {class_acc:<12.4f}")
            else:
                logger.info(f"{class_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {self.support[i]:<12,} {'N/A':<12}")
        
        # Class distribution
        logger.info("\n📈 CLASS DISTRIBUTION")
        logger.info("-" * 100)
        logger.info(f"{'Class':<20} {'Samples':<12} {'Percentage':<12} {'Distribution':<50}")
        logger.info("-" * 100)
        
        for i, class_name in enumerate(self.class_names):
            if self.support[i] > 0:
                percentage = (self.support[i] / self.total_samples) * 100
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                logger.info(f"{class_name:<20} {self.support[i]:<12,} {percentage:<12.2f}% {bar}")
            else:
                logger.info(f"{class_name:<20} {self.support[i]:<12,} {'0.00':<12}% ")
        
        # Misclassification analysis
        logger.info("\n❌ MISCLASSIFICATION ANALYSIS")
        logger.info("-" * 100)
        
        for i, true_class in enumerate(self.class_names):
            if self.support[i] == 0:
                continue
                
            misclassified = self.cm[i, :].sum() - self.cm[i, i]
            if misclassified > 0:
                logger.info(f"\n{true_class} (Total: {self.support[i]:,}, Correct: {self.cm[i,i]:,}, "
                          f"Misclassified: {misclassified:,})")
                
                for j, pred_class in enumerate(self.class_names):
                    if i != j and self.cm[i, j] > 0:
                        percentage = (self.cm[i, j] / self.support[i]) * 100
                        logger.info(f"  → Predicted as {pred_class}: {self.cm[i,j]:,} "
                                  f"({percentage:.2f}%)")
        
        # Best and worst performing classes
        logger.info("\n🏆 PERFORMANCE RANKING")
        logger.info("-" * 100)
        
        # Sort by F1-score
        valid_classes = [(i, name, f1, sup) for i, (name, f1, sup) in 
                        enumerate(zip(self.class_names, self.f1_score, self.support)) if sup > 0]
        valid_classes.sort(key=lambda x: x[2], reverse=True)
        
        logger.info("\nBest Performing Classes (by F1-Score):")
        for i, (idx, name, f1, sup) in enumerate(valid_classes[:3]):
            logger.info(f"  {i+1}. {name}: F1={f1:.4f} (Support: {sup:,})")
        
        if len(valid_classes) > 3:
            logger.info("\nWorst Performing Classes (by F1-Score):")
            for i, (idx, name, f1, sup) in enumerate(valid_classes[-3:]):
                logger.info(f"  {len(valid_classes)-i}. {name}: F1={f1:.4f} (Support: {sup:,})")
        
        logger.info("\n" + "="*100)
    
    def save_detailed_csv(self):
        """Save detailed metrics to CSV"""
        # Per-class metrics
        df_class = pd.DataFrame({
            'Class': self.class_names,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1-Score': self.f1_score,
            'Support': self.support,
            'Accuracy': [self.cm[i, i] / self.support[i] if self.support[i] > 0 else 0 
                        for i in range(len(self.class_names))]
        })
        
        csv_path = self.output_dir / 'detailed_per_class_metrics.csv'
        df_class.to_csv(csv_path, index=False)
        logger.info(f"\n💾 Saved per-class metrics: {csv_path}")
        
        # Overall metrics
        df_overall = pd.DataFrame({
            'Metric': ['Overall Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score',
                      'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score', 
                      'Total Samples', 'Correct Predictions'],
            'Value': [self.accuracy, self.macro_precision, self.macro_recall, self.macro_f1,
                     self.weighted_precision, self.weighted_recall, self.weighted_f1,
                     self.total_samples, self.correct_predictions]
        })
        
        csv_path = self.output_dir / 'detailed_overall_metrics.csv'
        df_overall.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved overall metrics: {csv_path}")
        
        # Confusion matrix as CSV
        df_cm = pd.DataFrame(self.cm, index=self.class_names, columns=self.class_names)
        csv_path = self.output_dir / 'confusion_matrix.csv'
        df_cm.to_csv(csv_path)
        logger.info(f"💾 Saved confusion matrix: {csv_path}")
    
    def create_visualizations(self):
        """Create detailed visualizations"""
        plt.style.use('default')
        
        # 1. Enhanced Confusion Matrix with percentages
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Confusion Matrix (Percentages)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        img_path = self.output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved normalized confusion matrix: {img_path}")
        
        # 2. Per-class metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        valid_indices = [i for i, s in enumerate(self.support) if s > 0]
        valid_classes = [self.class_names[i] for i in valid_indices]
        
        metrics = {
            'Precision': [self.precision[i] for i in valid_indices],
            'Recall': [self.recall[i] for i in valid_indices],
            'F1-Score': [self.f1_score[i] for i in valid_indices],
            'Support': [self.support[i] for i in valid_indices]
        }
        
        x = np.arange(len(valid_classes))
        width = 0.25
        
        # Precision, Recall, F1 comparison
        ax = axes[0, 0]
        ax.bar(x - width, metrics['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
        ax.set_xlabel('Classes', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Per-Class Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # F1-Score ranking
        ax = axes[0, 1]
        sorted_idx = np.argsort([metrics['F1-Score'][i] for i in range(len(valid_classes))])
        sorted_classes = [valid_classes[i] for i in sorted_idx]
        sorted_f1 = [metrics['F1-Score'][i] for i in sorted_idx]
        
        colors = plt.cm.RdYlGn([f1 for f1 in sorted_f1])
        ax.barh(sorted_classes, sorted_f1, color=colors, alpha=0.8)
        ax.set_xlabel('F1-Score', fontweight='bold')
        ax.set_title('Classes Ranked by F1-Score', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([0, 1.1])
        
        # Add values on bars
        for i, (c, f1) in enumerate(zip(sorted_classes, sorted_f1)):
            ax.text(f1 + 0.02, i, f'{f1:.3f}', va='center')
        
        # Class distribution
        ax = axes[1, 0]
        colors_dist = plt.cm.Set3(np.linspace(0, 1, len(valid_classes)))
        wedges, texts, autotexts = ax.pie(metrics['Support'], labels=valid_classes, 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_dist)
        ax.set_title('Class Distribution', fontweight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Support count
        ax = axes[1, 1]
        ax.bar(valid_classes, metrics['Support'], color=colors_dist, alpha=0.8)
        ax.set_xlabel('Classes', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Sample Count per Class', fontweight='bold')
        ax.set_xticklabels(valid_classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (c, s) in enumerate(zip(valid_classes, metrics['Support'])):
            ax.text(i, s + max(metrics['Support'])*0.02, f"{s:,}", 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        img_path = self.output_dir / 'detailed_metrics_visualization.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved detailed metrics visualization: {img_path}")
        
        # 3. Error analysis heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create error matrix (excluding diagonal)
        error_matrix = self.cm.copy()
        np.fill_diagonal(error_matrix, 0)
        
        # Normalize by row
        error_normalized = error_matrix.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        error_normalized = np.nan_to_num(error_normalized)
        
        # Create annotations with both count and percentage
        annot = np.empty_like(error_matrix, dtype=object)
        for i in range(error_matrix.shape[0]):
            for j in range(error_matrix.shape[1]):
                count = error_matrix[i, j]
                pct = error_normalized[i, j] * 100
                if count > 0:
                    annot[i, j] = f'{count}\n({pct:.1f}%)'
                else:
                    annot[i, j] = ''
        
        sns.heatmap(error_normalized, annot=annot, fmt='', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Error Rate'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Misclassification Patterns (Errors Only)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        img_path = self.output_dir / 'error_analysis_heatmap.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved error analysis heatmap: {img_path}")
    
    def save_json_report(self):
        """Save complete report as JSON"""
        report = {
            'overall_metrics': {
                'accuracy': float(self.accuracy),
                'macro_precision': float(self.macro_precision),
                'macro_recall': float(self.macro_recall),
                'macro_f1': float(self.macro_f1),
                'weighted_precision': float(self.weighted_precision),
                'weighted_recall': float(self.weighted_recall),
                'weighted_f1': float(self.weighted_f1),
                'total_samples': int(self.total_samples),
                'correct_predictions': int(self.correct_predictions),
                'incorrect_predictions': int(self.total_samples - self.correct_predictions)
            },
            'per_class_metrics': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            report['per_class_metrics'][class_name] = {
                'precision': float(self.precision[i]),
                'recall': float(self.recall[i]),
                'f1_score': float(self.f1_score[i]),
                'support': int(self.support[i]),
                'accuracy': float(self.cm[i, i] / self.support[i] if self.support[i] > 0 else 0),
                'misclassifications': {}
            }
            
            # Add misclassification details
            for j, pred_class in enumerate(self.class_names):
                if i != j and self.cm[i, j] > 0:
                    report['per_class_metrics'][class_name]['misclassifications'][pred_class] = {
                        'count': int(self.cm[i, j]),
                        'percentage': float((self.cm[i, j] / self.support[i]) * 100 if self.support[i] > 0 else 0)
                    }
        
        json_path = self.output_dir / 'detailed_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Saved JSON report: {json_path}")
    
    def generate_complete_report(self):
        """Generate complete detailed report"""
        self.print_detailed_report()
        self.save_detailed_csv()
        self.create_visualizations()
        self.save_json_report()
        
        logger.info("\n✅ Complete detailed evaluation report generated!")
        logger.info(f"📁 All files saved to: {self.output_dir}")


def main():
    """Main function to generate detailed report from existing confusion matrix"""
    # Configuration
    evaluation_dir = Path(r"..\..\outputs\trained_models\engagement_organized_full\evaluation")
    
    # Read existing confusion matrix from the log output
    confusion_matrix = np.array([
        [21881, 399, 423, 0],
        [242, 24316, 204, 0],
        [411, 490, 16531, 0],
        [0, 0, 0, 0]
    ])
    
    # Class names (adjust based on your dataset)
    class_names = ['disengaged', 'engaged', 'moderately-engaged', 'unknown']
    
    # Create output directory for detailed report
    output_dir = evaluation_dir / 'detailed_report'
    
    # Generate detailed report
    logger.info("Starting detailed evaluation report generation...")
    report = DetailedEvaluationReport(confusion_matrix, class_names, output_dir)
    report.generate_complete_report()


if __name__ == "__main__":
    main()