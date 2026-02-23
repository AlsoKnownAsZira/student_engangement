"""
Quick Demo: Generate Confusion Matrix Analysis

This script demonstrates how to use the confusion matrix generator
with your existing data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_confusion_matrix import ConfusionMatrixGenerator

print("="*80)
print("CONFUSION MATRIX DEMO")
print("="*80)

# Option 1: Using only predictions (distribution analysis)
print("\n📊 OPTION 1: Prediction Distribution Analysis (No Ground Truth)")
print("-"*80)

try:
    # Check if sav2.csv exists
    if Path('sav2.csv').exists():
        print("✓ Found sav2.csv")
        
        generator = ConfusionMatrixGenerator(
            predictions_csv='sav2.csv',
            ground_truth_csv=None,
            output_dir='demo_analysis_no_gt'
        )
        
        print("\nGenerating distribution analysis...")
        generator.plot_prediction_distribution()
        
        print(f"\n✅ Done! Check the 'demo_analysis_no_gt' folder for results.")
        print(f"   - distribution_comparison.png: Shows prediction distribution")
        
    else:
        print("❌ sav2.csv not found in current directory")
        print("   Please run this from the phase4_pipeline folder")

except Exception as e:
    print(f"❌ Error: {e}")


# Option 2: Create a synthetic ground truth for demonstration
print("\n\n📊 OPTION 2: Creating Synthetic Ground Truth for Demo")
print("-"*80)
print("This will create fake ground truth data to demonstrate the confusion matrix.")

try:
    if Path('sav2.csv').exists():
        # Load predictions
        df = pd.read_csv('sav2.csv')
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Create synthetic ground truth (90% same as prediction, 10% different for demo)
        np.random.seed(42)
        
        gt_df = df[['frame', 'track_id']].copy()
        gt_df['ground_truth'] = df['engagement_level'].copy()
        
        # Add some random "errors" for demonstration
        n_samples = len(gt_df)
        n_errors = int(n_samples * 0.1)  # 10% errors
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        
        classes = ['high', 'medium', 'low']
        for idx in error_indices:
            current = gt_df.loc[idx, 'ground_truth']
            # Pick a different class
            other_classes = [c for c in classes if c != current]
            gt_df.loc[idx, 'ground_truth'] = np.random.choice(other_classes)
        
        # Save synthetic ground truth
        gt_path = 'demo_ground_truth.csv'
        gt_df.to_csv(gt_path, index=False)
        print(f"✓ Created synthetic ground truth: {gt_path}")
        print(f"  (90% matches predictions, 10% different for demo purposes)")
        
        # Generate full confusion matrix analysis
        print("\nGenerating full confusion matrix analysis...")
        generator = ConfusionMatrixGenerator(
            predictions_csv='sav2.csv',
            ground_truth_csv=gt_path,
            output_dir='demo_analysis_with_gt'
        )
        
        generator.generate_complete_analysis()
        
        print(f"\n✅ Done! Check the 'demo_analysis_with_gt' folder for:")
        print(f"   - confusion_matrix.png: Raw count confusion matrix")
        print(f"   - confusion_matrix_normalized.png: Percentage confusion matrix")
        print(f"   - per_class_accuracy.png: Accuracy for each class")
        print(f"   - distribution_comparison.png: Predictions vs ground truth")
        print(f"   - classification_report.txt: Detailed metrics")
        print(f"   - per_class_metrics.csv: Metrics in CSV format")
        
    else:
        print("❌ sav2.csv not found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\n📖 For real analysis, you need actual ground truth data.")
print("   See CONFUSION_MATRIX_GUIDE.md for more information.")
print("\n💡 To create a ground truth template:")
print("   python create_ground_truth_template.py --predictions sav2.csv --sample 100")
