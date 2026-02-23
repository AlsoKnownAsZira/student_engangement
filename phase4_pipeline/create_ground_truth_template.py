"""
Create Sample Ground Truth Template
This script helps you create a ground truth template from your predictions CSV
"""

import pandas as pd
import argparse
from pathlib import Path


def create_ground_truth_template(predictions_csv, output_csv='ground_truth_template.csv', sample_size=None):
    """
    Create a ground truth template from predictions
    
    Args:
        predictions_csv: Path to predictions CSV
        output_csv: Output path for ground truth template
        sample_size: Number of samples to include (None = all)
    """
    print(f"📂 Loading predictions from {predictions_csv}...")
    df = pd.read_csv(predictions_csv)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Extract relevant columns
    columns_to_keep = []
    
    if 'frame' in df.columns:
        columns_to_keep.append('frame')
    if 'track_id' in df.columns:
        columns_to_keep.append('track_id')
    
    # Add prediction as reference
    if 'engagement_level' in df.columns:
        columns_to_keep.append('engagement_level')
    
    # Add bounding box for reference (optional)
    if all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2']):
        columns_to_keep.extend(['x1', 'y1', 'x2', 'y2'])
    
    # Create template
    template = df[columns_to_keep].copy()
    
    # Rename engagement_level to show it's a prediction
    if 'engagement_level' in template.columns:
        template = template.rename(columns={'engagement_level': 'predicted_label'})
    
    # Add empty column for ground truth
    template['ground_truth'] = ''
    
    # Sample if requested
    if sample_size and sample_size < len(template):
        template = template.sample(n=sample_size, random_state=42).sort_index()
        print(f"📊 Sampled {sample_size} out of {len(df)} samples")
    else:
        sample_size = len(template)
        print(f"📊 Using all {sample_size} samples")
    
    # Save
    output_path = Path(output_csv)
    template.to_csv(output_path, index=False)
    
    print(f"✅ Ground truth template saved to {output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Open {output_path}")
    print(f"   2. Fill in the 'ground_truth' column with actual labels: high, medium, or low")
    print(f"   3. Use the filled CSV with: python generate_confusion_matrix.py --predictions {predictions_csv} --ground_truth {output_csv}")
    
    # Show sample
    print(f"\n📋 Template preview (first 5 rows):")
    print(template.head())
    
    return template


def main():
    parser = argparse.ArgumentParser(description='Create ground truth template from predictions')
    
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--output', type=str, default='ground_truth_template.csv',
                       help='Output path for ground truth template')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of samples to include (default: all)')
    
    args = parser.parse_args()
    
    create_ground_truth_template(
        predictions_csv=args.predictions,
        output_csv=args.output,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()
