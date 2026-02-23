# Confusion Matrix Generation Guide

## Overview
The `generate_confusion_matrix.py` script generates comprehensive confusion matrix analysis for engagement classification (high, medium, low).

## Features
✅ Generates confusion matrix (both raw counts and normalized percentages)  
✅ Classification report with precision, recall, F1-score  
✅ Per-class accuracy visualization  
✅ Distribution comparison plots  
✅ Works with or without ground truth data  

## Requirements
Make sure you have the necessary packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Option 1: With Ground Truth Data (Full Analysis)
If you have ground truth labels for your predictions:

```bash
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth ground_truth.csv
```

**Ground truth CSV format requirements:**
- Must have matching keys with predictions CSV (`frame` and/or `track_id`)
- Must have one of these column names for labels: `engagement_level`, `label`, or `true_label`
- Labels should be: `high`, `medium`, or `low`

Example ground truth CSV:
```csv
frame,track_id,engagement_level
0,1,medium
0,2,low
0,3,low
1,1,medium
...
```

### Option 2: Without Ground Truth (Distribution Only)
If you only want to analyze prediction distribution:

```bash
python generate_confusion_matrix.py --predictions sav2.csv
```

This will generate:
- Prediction distribution plots
- Summary statistics
- No confusion matrix (since we need ground truth to compare)

### Option 3: Custom Output Directory
```bash
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth gt.csv --output my_results
```

## Output Files

When ground truth is provided, the script generates:

📊 **Visual Outputs:**
1. `confusion_matrix.png` - Raw count confusion matrix
2. `confusion_matrix_normalized.png` - Percentage-based confusion matrix
3. `per_class_accuracy.png` - Bar chart of accuracy per class
4. `distribution_comparison.png` - Predictions vs ground truth distribution

📄 **Text Outputs:**
1. `classification_report.txt` - Detailed metrics report
2. `per_class_metrics.csv` - Per-class precision, recall, F1-score

## Example Workflow

### Step 1: Prepare Your Data
Make sure your predictions CSV (e.g., `sav2.csv`) has the engagement predictions.

### Step 2: Prepare Ground Truth (if available)
Create a CSV file with the true labels for your predictions. Match the format shown above.

### Step 3: Run the Script
```bash
cd phase4_pipeline
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth ground_truth.csv
```

### Step 4: View Results
Check the `confusion_matrix_analysis/` folder for all generated outputs.

## Understanding the Outputs

### Confusion Matrix
- **Rows**: True labels (what the engagement actually was)
- **Columns**: Predicted labels (what the model predicted)
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

### Classification Report
- **Precision**: Of all predictions for a class, how many were correct?
- **Recall**: Of all actual instances of a class, how many were detected?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual samples for each class

### Per-Class Accuracy
Shows how well the model performs on each engagement level separately.

## Troubleshooting

### "Cannot find common merge keys"
- Ensure both CSVs have `frame` and/or `track_id` columns
- Check column names match exactly (case-sensitive)

### "No matching samples found"
- Verify that frame and track_id values match between files
- Check if there's any data type mismatch

### "Ground truth CSV must have one of..."
- Rename your label column to `engagement_level`, `label`, or `true_label`

## Quick Test
To test if the script works with your current predictions (without ground truth):
```bash
python generate_confusion_matrix.py --predictions sav2.csv
```

This will show you the prediction distribution and verify the script runs correctly.
