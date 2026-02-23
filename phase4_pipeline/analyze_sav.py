"""
Analyze sav.csv and generate confusion matrix
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_confusion_matrix import ConfusionMatrixGenerator

print("="*80)
print("ANALYZING sav.csv")
print("="*80)

# Check if file exists
if not Path('sav.csv').exists():
    print("❌ sav.csv not found in current directory")
    exit(1)

# Show statistics
print("\n📊 Loading sav.csv...")
df = pd.read_csv('sav.csv')
df.columns = df.columns.str.strip()

print("\n" + "-"*80)
print("DATA STATISTICS")
print("-"*80)
print(f"Total samples: {len(df)}")
if 'frame' in df.columns:
    print(f"Unique frames: {df['frame'].nunique()}")
if 'track_id' in df.columns:
    print(f"Unique students (track_id): {df['track_id'].nunique()}")

if 'engagement_level' in df.columns:
    print("\n" + "-"*80)
    print("ENGAGEMENT LEVEL DISTRIBUTION:")
    print("-"*80)
    dist = df['engagement_level'].value_counts()
    for level, count in dist.items():
        pct = (count / len(df)) * 100
        print(f"  {level:10s}: {count:5d} ({pct:5.2f}%)")
else:
    print("\n⚠️  No 'engagement_level' column found")
    print(f"Columns: {df.columns.tolist()}")

# Option 1: Distribution analysis (no ground truth)
print("\n" + "="*80)
print("GENERATING DISTRIBUTION ANALYSIS")
print("="*80)

try:
    generator = ConfusionMatrixGenerator(
        predictions_csv='sav.csv',
        ground_truth_csv=None,
        output_dir='sav_analysis'
    )
    
    generator.plot_prediction_distribution()
    
    print(f"\n✅ Distribution analysis complete!")
    print(f"   Results saved to: sav_analysis/")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Option 2: Create synthetic ground truth for demo
print("\n" + "="*80)
print("CREATING DEMO WITH SYNTHETIC GROUND TRUTH")
print("="*80)

try:
    # Create synthetic ground truth (90% same as prediction, 10% different)
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
        other_classes = [c for c in classes if c != current]
        gt_df.loc[idx, 'ground_truth'] = np.random.choice(other_classes)
    
    # Save synthetic ground truth
    gt_path = 'sav_demo_ground_truth.csv'
    gt_df.to_csv(gt_path, index=False)
    print(f"✓ Created synthetic ground truth: {gt_path}")
    
    # Generate full confusion matrix analysis
    print("\nGenerating full confusion matrix analysis...")
    generator = ConfusionMatrixGenerator(
        predictions_csv='sav.csv',
        ground_truth_csv=gt_path,
        output_dir='sav_analysis_with_demo_gt'
    )
    
    generator.generate_complete_analysis()
    
    print(f"\n✅ Full analysis complete!")
    print(f"   Results saved to: sav_analysis_with_demo_gt/")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\n📁 Check these folders:")
print("   - sav_analysis/ (distribution only)")
print("   - sav_analysis_with_demo_gt/ (full confusion matrix with demo data)")
print("\n💡 For real analysis with your own ground truth:")
print("   python generate_confusion_matrix.py --predictions sav.csv --ground_truth YOUR_GT.csv")
