# Comparison: sav.csv vs sav2.csv Analysis

## 📊 Data Comparison

### sav.csv
- **Total Samples**: 5,166
- **Frames**: 300
- **Students Tracked**: 73
- **Distribution**:
  - High (Engaged): 2,345 (45.39%)
  - Low (Disengaged): 1,535 (29.71%)
  - Medium (Moderately): 1,286 (24.89%)

### sav2.csv
- **Total Samples**: 2,873
- **Frames**: 300
- **Students Tracked**: 23
- **Distribution**:
  - Low (Disengaged): 1,675 (58.30%)
  - Medium (Moderately): 787 (27.39%)
  - High (Engaged): 411 (14.31%)

## 🔍 Key Differences

1. **sav.csv has 80% MORE data** (5,166 vs 2,873 samples)
2. **sav.csv tracks 3x MORE students** (73 vs 23 students)
3. **Different engagement patterns**:
   - sav.csv: More "High" engagement (45% vs 14%)
   - sav2.csv: More "Low" engagement (58% vs 30%)

## 📁 Generated Outputs

### For sav.csv (in `sav_analysis_with_demo_gt/`)
- ✅ confusion_matrix.png
- ✅ confusion_matrix_normalized.png
- ✅ per_class_accuracy.png
- ✅ distribution_comparison.png
- ✅ classification_report.txt
- ✅ per_class_metrics.csv

### For sav2.csv (in `demo_analysis_with_gt/`)
- ✅ confusion_matrix.png
- ✅ confusion_matrix_normalized.png
- ✅ per_class_accuracy.png
- ✅ distribution_comparison.png
- ✅ classification_report.txt
- ✅ per_class_metrics.csv

## 🚀 Using with Real Ground Truth

For both files, you can generate confusion matrices with your actual ground truth:

```bash
# For sav.csv
python generate_confusion_matrix.py --predictions sav.csv --ground_truth your_gt.csv --output sav_real_analysis

# For sav2.csv
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth your_gt.csv --output sav2_real_analysis

# For sav3.csv (if you have it)
python generate_confusion_matrix.py --predictions sav3.csv --ground_truth your_gt.csv --output sav3_real_analysis
```

## 📝 Notes

- Demo results use **90% synthetic accuracy** for demonstration
- For real evaluation, you need actual ground truth labels
- Both analyses show the same tools work with different datasets
- The script automatically handles different data sizes and distributions
