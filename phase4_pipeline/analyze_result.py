"""
Comprehensive Analysis Script for Phase 4 Results
Generates confusion matrix, visualizations, and detailed metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# For analysis - use these if available
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not found. Install with: pip install scikit-learn")


class ResultsAnalyzer:
    """Analyze Phase 4 pipeline results"""
    
    # Class mappings
    CLASS_MAPPING = {
        'high': 'engaged',
        'medium': 'moderately-engaged',
        'low': 'disengaged'
    }
    
    CLASS_ORDER_NEW = ['engaged', 'moderately-engaged', 'disengaged']
    CLASS_ORDER_OLD = ['high', 'medium', 'low']
    
    def __init__(self, csv_path, ground_truth_csv=None, output_dir='analysis'):
        """Initialize analyzer"""
        self.csv_path = Path(csv_path)
        self.ground_truth_csv = ground_truth_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Detect class convention
        classes = self.df['engagement_level'].unique()
        if any(c in ['high', 'medium', 'low'] for c in classes):
            self.convention = 'old'
            self.class_order = self.CLASS_ORDER_OLD
            print("📊 Detected: OLD class names (high/medium/low)")
        else:
            self.convention = 'new'
            self.class_order = self.CLASS_ORDER_NEW
            print("📊 Detected: NEW class names (engaged/moderately-engaged/disengaged)")
        
        # Load ground truth if provided
        self.has_ground_truth = False
        if ground_truth_csv and Path(ground_truth_csv).exists():
            self.gt_df = pd.read_csv(ground_truth_csv)
            self.has_ground_truth = True
            print(f"✓ Ground truth loaded from {ground_truth_csv}")
        
        print(f"✓ Loaded {len(self.df)} detections from {self.df['frame'].nunique()} frames")
        print(f"✓ Unique students: {self.df['track_id'].nunique()}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        stats = {
            'total_frames': int(self.df['frame'].nunique()),
            'total_detections': len(self.df),
            'unique_students': int(self.df['track_id'].nunique()),
            'avg_students_per_frame': float(len(self.df) / self.df['frame'].nunique()),
            'avg_detection_confidence': float(self.df['detection_conf'].mean()),
            'avg_classification_confidence': float(self.df['engagement_score'].mean()),
            'min_classification_confidence': float(self.df['engagement_score'].min()),
            'max_classification_confidence': float(self.df['engagement_score'].max()),
        }
        
        # Engagement distribution
        dist = self.df['engagement_level'].value_counts()
        stats['engagement_distribution'] = {
            str(k): int(v) for k, v in dist.items()
        }
        stats['engagement_distribution_pct'] = {
            str(k): float(v/len(self.df)*100) for k, v in dist.items()
        }
        
        # Print stats
        print(f"\nTotal Frames: {stats['total_frames']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Unique Students: {stats['unique_students']}")
        print(f"Avg Students/Frame: {stats['avg_students_per_frame']:.2f}")
        print(f"\nDetection Confidence: {stats['avg_detection_confidence']:.3f}")
        print(f"Classification Confidence: {stats['avg_classification_confidence']:.3f}")
        print(f"  Min: {stats['min_classification_confidence']:.3f}")
        print(f"  Max: {stats['max_classification_confidence']:.3f}")
        
        print(f"\nEngagement Distribution:")
        for level in self.class_order:
            if level in stats['engagement_distribution']:
                count = stats['engagement_distribution'][level]
                pct = stats['engagement_distribution_pct'][level]
                print(f"  {level}: {count} ({pct:.1f}%)")
        
        # Save stats
        stats_path = self.output_dir / 'summary_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Saved statistics to {stats_path}")
        
        return stats
    
    def plot_engagement_distribution(self):
        """Plot engagement distribution"""
        print("\n📊 Generating engagement distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        dist = self.df['engagement_level'].value_counts()
        dist = dist.reindex(self.class_order, fill_value=0)
        
        colors = {
            'high': '#2ecc71', 'engaged': '#2ecc71',
            'medium': '#f39c12', 'moderately-engaged': '#f39c12',
            'low': '#e74c3c', 'disengaged': '#e74c3c'
        }
        
        bar_colors = [colors.get(c, '#95a5a6') for c in dist.index]
        
        axes[0].bar(range(len(dist)), dist.values, color=bar_colors)
        axes[0].set_xticks(range(len(dist)))
        axes[0].set_xticklabels(dist.index, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Engagement Level Distribution (Count)')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for i, v in enumerate(dist.values):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # Percentage pie chart
        axes[1].pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                   colors=bar_colors, startangle=90)
        axes[1].set_title('Engagement Level Distribution (%)')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'engagement_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to {plot_path}")
    
    def plot_temporal_analysis(self):
        """Plot engagement over time"""
        print("\n📊 Generating temporal analysis plots...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Engagement count per frame
        frame_engagement = self.df.groupby(['frame', 'engagement_level']).size().unstack(fill_value=0)
        frame_engagement = frame_engagement.reindex(columns=self.class_order, fill_value=0)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        frame_engagement.plot(kind='area', stacked=True, ax=axes[0], color=colors, alpha=0.7)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Number of Students')
        axes[0].set_title('Engagement Levels Over Time (Stacked)')
        axes[0].legend(title='Engagement Level', loc='upper left')
        axes[0].grid(alpha=0.3)
        
        # Percentage view
        frame_engagement_pct = frame_engagement.div(frame_engagement.sum(axis=1), axis=0) * 100
        frame_engagement_pct.plot(kind='area', stacked=True, ax=axes[1], color=colors, alpha=0.7)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Engagement Levels Over Time (Percentage)')
        axes[1].legend(title='Engagement Level', loc='upper left')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'temporal_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to {plot_path}")
    
    def analyze_per_student(self):
        """Analyze engagement per student"""
        print("\n📊 Analyzing per-student engagement...")
        
        # Aggregate per student
        student_stats = self.df.groupby('track_id').agg({
            'engagement_score': ['mean', 'std', 'min', 'max'],
            'engagement_level': lambda x: x.mode()[0] if len(x) > 0 else None,
            'frame': 'count'
        }).reset_index()
        
        student_stats.columns = ['track_id', 'avg_confidence', 'std_confidence', 
                                'min_confidence', 'max_confidence', 'dominant_level', 'appearances']
        
        # Sort by appearances
        student_stats = student_stats.sort_values('appearances', ascending=False)
        
        # Save to CSV
        csv_path = self.output_dir / 'per_student_analysis.csv'
        student_stats.to_csv(csv_path, index=False)
        print(f"✓ Saved per-student analysis to {csv_path}")
        
        return student_stats
    
    def generate_full_report(self):
        """Generate complete analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # All analyses
        stats = self.generate_summary_statistics()
        self.plot_engagement_distribution()
        self.plot_temporal_analysis()
        student_stats = self.analyze_per_student()
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print(f"✓ All outputs saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Analyze Phase 4 pipeline results')
    
    parser.add_argument('--csv', type=str, required=True, help='Path to pipeline output CSV')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ResultsAnalyzer(csv_path=args.csv, output_dir=args.output)
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()