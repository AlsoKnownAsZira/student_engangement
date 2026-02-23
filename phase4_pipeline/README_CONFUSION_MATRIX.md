# Confusion Matrix Generator - Quick Start

## ✅ What I Created For You

I've created a complete confusion matrix generation system with 3 scripts:

### 1. **generate_confusion_matrix.py** (Main Script)
   - Generates confusion matrices for high, medium, low engagement levels
   - Creates both raw count and normalized percentage matrices
   - Produces classification reports with precision, recall, F1-score
   - Generates per-class accuracy visualizations
   - Works with or without ground truth data

### 2. **create_ground_truth_template.py** (Helper Script)
   - Creates a template CSV from your predictions for easy manual labeling
   - Useful if you need to manually label ground truth data

### 3. **demo_confusion_matrix.py** (Demo Script)
   - Demonstrates the system with synthetic data
   - Already run successfully! ✅

---

## 📊 Demo Results Generated

Check these folders:
- **demo_analysis_no_gt/** - Prediction distribution (no ground truth)
- **demo_analysis_with_gt/** - Full confusion matrix analysis with synthetic ground truth

Files in `demo_analysis_with_gt/`:
- `confusion_matrix.png` - Raw count confusion matrix
- `confusion_matrix_normalized.png` - Percentage-based confusion matrix  
- `per_class_accuracy.png` - Bar chart of accuracy per class
- `distribution_comparison.png` - Predictions vs ground truth distribution
- `classification_report.txt` - Detailed metrics report
- `per_class_metrics.csv` - Metrics in CSV format

---

## 🚀 How to Use With Your Data

### Option 1: If You HAVE Ground Truth Labels

```bash
cd phase4_pipeline
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth your_ground_truth.csv
```

**Ground truth CSV requirements:**
- Must have `frame` and `track_id` columns (to match with predictions)
- Must have one of these columns with labels: `engagement_level`, `label`, or `true_label`
- Labels must be: `high`, `medium`, or `low`

Example ground truth format:
```csv
frame,track_id,engagement_level
0,1,medium
0,2,low
0,3,low
1,1,high
...
```

### Option 2: If You DON'T HAVE Ground Truth Yet

#### **A. Create a template for manual labeling:**
```bash
# Create template with 100 random samples
python create_ground_truth_template.py --predictions sav2.csv --sample 100 --output my_ground_truth.csv

# Then manually fill in the 'ground_truth' column in my_ground_truth.csv
# After filling it, run:
python generate_confusion_matrix.py --predictions sav2.csv --ground_truth my_ground_truth.csv
```

#### **B. Only analyze prediction distribution:**
```bash
python generate_confusion_matrix.py --predictions sav2.csv
```
This shows distribution but can't generate confusion matrix without ground truth.

---

## 📋 What You Need To Do Next

1. **If you have ground truth data:**
   - Prepare your ground truth CSV with the format shown above
   - Run: `python generate_confusion_matrix.py --predictions sav2.csv --ground_truth YOUR_FILE.csv`

2. **If you need to create ground truth:**
   - Create template: `python create_ground_truth_template.py --predictions sav2.csv --sample 100`
   - Manually label the samples in the template
   - Run the confusion matrix generator with your labeled data

3. **View the results:**
   - Check the `confusion_matrix_analysis/` folder for all outputs
   - The confusion matrix shows:
     - **Rows** = True labels (what engagement actually was)
     - **Columns** = Predicted labels (what model predicted)
     - **Diagonal** = Correct predictions ✅
     - **Off-diagonal** = Misclassifications ❌

---

## 🎯 Understanding Confusion Matrix

```
                    Predicted
                High  Medium  Low
True   High     [90]    [5]   [5]    ← 90% correct for "high"
       Medium   [10]   [75]  [15]    ← 75% correct for "medium"  
       Low      [3]    [7]   [90]    ← 90% correct for "low"
```

Perfect predictions would have large numbers on the diagonal and zeros everywhere else.

---

## 📚 Additional Files Created

- **CONFUSION_MATRIX_GUIDE.md** - Detailed documentation
- **demo_ground_truth.csv** - Synthetic ground truth used in demo
- All demo output folders with example visualizations

---

## ❓ Questions?

- **Q: I don't have ground truth labels, how do I get them?**
  - A: Use `create_ground_truth_template.py` to create a template, then manually label samples

- **Q: Can I use different CSV files (not sav2.csv)?**
  - A: Yes! Use `--predictions YOUR_FILE.csv`

- **Q: What if columns don't match?**
  - A: The script automatically strips spaces and looks for 'frame' and 'track_id' columns

- **Q: Can I analyze all my CSV files (sav.csv, sav2.csv, sav3.csv)?**
  - A: Yes, run the script separately for each file or combine them first

---

**Next Step:** Prepare your ground truth data or create a template to start labeling! 🎯
