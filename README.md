# Person Tracking & Engagement Classification System

> **Status Update**: Phase 3 training COMPLETED! Model achieved **96.66% accuracy** with excellent per-class performance! 🎉🚀

Sistem deteksi dan tracking per-siswa dengan klasifikasi engagement level menggunakan YOLOv11.

## 📋 Deskripsi

Sistem ini mendeteksi setiap siswa dalam video, melakukan tracking dengan ID unik untuk setiap siswa, dan mengklasifikasikan engagement level mereka. Pendekatan ini **fokus ke individual siswa** sehingga lebih robust terhadap perubahan angle kamera dan layout ruangan jika dibandingkan dengan membaca keseluruhan frame.

## 🎯 Keuntungan vs Metode Lama

| Aspek | Metode Lama | Metode Baru |
|-------|------------|-------------|
| **Unit Analisis** | Seluruh frame | Per siswa |
| **Robustness** | Sensitif terhadap angle & layout | Robust, fokus ke person |
| **Detail** | Agregat kelas | Detail per siswa |
| **Generalisasi** | Terbatas pada setup tertentu | Generalisasi lebih baik |
| **Tracking** | Tidak ada | Ada tracking ID |

---

## 📊 Model Performance (Latest)

### Overall Metrics

- **Total Samples**: 64,897
- **Overall Accuracy**: **96.66%** ✨
- **Macro-averaged Precision**: 0.9664
- **Macro-averaged Recall**: 0.9647
- **Macro-averaged F1-Score**: 0.9655

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Accuracy |
|-------|-----------|--------|----------|---------|----------|
| **engaged/high** | 0.9647 | 0.9820 | **0.9733** 🏆 | 24,762 | 98.20% |
| **disengaged/low** | 0.9710 | 0.9638 | **0.9674** | 22,703 | 96.38% |
| **moderately-engaged/medium** | 0.9635 | 0.9483 | **0.9558** | 17,432 | 94.83% |

### Class Distribution

```
engaged/high             : 38.16% (24,762 samples) ███████████████████
disengaged/low          : 34.98% (22,703 samples) █████████████████
moderately-engaged/medium  : 26.86% (17,432 samples) █████████████
```

### Misclassification Analysis

- **disengaged/low**: 822 misclassifications (3.62%)
  - → engaged: 399 (1.76%)
  - → moderately-engaged: 423 (1.86%)
  
- **engaged/high**: 446 misclassifications (1.80%)
  - → disengaged: 242 (0.98%)
  - → moderately-engaged: 204 (0.82%)
  
- **moderately-engaged/medium**: 901 misclassifications (5.17%)
  - → disengaged: 411 (2.36%)
  - → engaged: 490 (2.81%)

**Key Insight**: Kelas "engaged/high" memiliki performa terbaik (F1=0.9733), sementara "moderately-engaged/medium" sedikit lebih challenging karena berada di antara dua ekstrem.

---

## 📁 Struktur Project

```
person-tracking-engagement/
├── config.py                    # Konfigurasi utama
├── requirements.txt             # Dependencies
│
├── phase1_poc/                  # ✅ FASE 1: Proof of Concept
│   ├── detect_track.py          # Detection + tracking + scoring
│   ├── visualizer.py            # Visualisasi hasil
│   └── run_poc.py              # Main runner
│
├── phase2_dataset/              # ✅ FASE 2: Dataset Preparation
│   ├── person_crops_organized/  # 671,649 person crops (balanced!)
│   │   ├── train/              # Training set
│   │   ├── val/                # Validation set
│   │   └── test/               # Test set
│   └── crop_organized_dataset.py  # Main cropping tool
│
├── phase3_training/             # ✅ FASE 3: Training (COMPLETED!)
│   ├── train_classifier.py      # Train classifier
│   ├── evaluate.py              # Evaluasi model
│   ├── detailed_evaluation_report.py  # Detailed reporting
│   └── runs/                    # Training runs
│       └── engagement_organized_full/  # Best model: 96.66%
│           └── weights/
│               └── best.pt      # Production-ready model
│
├── phase4_pipeline/             # 🔜 FASE 4: Full Pipeline
│   └── full_pipeline.py         # Pipeline lengkap (coming soon)
│
├── utils/                       # Utilities
│   ├── video_utils.py
│   ├── metrics.py
│   └── logger.py
│
└── outputs/                     # Output directory
    ├── poc_results/
    ├── person_crops/
    └── trained_models/
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Buat virtual environment
python -m venv venv-person-tracking

# Aktivasi
# Windows PowerShell:
.\venv-person-tracking\Scripts\Activate.ps1
# Linux/Mac:
source venv-person-tracking/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi Path

Edit `config.py` dan sesuaikan path:

```python
# Dataset video OUC-CGE
VIDEO_DATASET_ROOT = r"F:\OUC-CGE dataset"

# Working directory
WORK_DIR = r"D:\kuliah\Skripsi\person-tracking-engagement"
```

---

## 📖 Workflow Detail

### ✅ FASE 1: Proof of Concept (COMPLETED)

**Goal**: Validasi konsep person-level detection + tracking + scoring

```bash
cd phase1_poc
python run_poc.py --video_path "path/to/video.mp4" --limit_frames 100
```

**Output**: Video dengan bounding box, track ID, dan engagement score per person

**Status**: ✅ Konsep terbukti berhasil!

---

### ✅ FASE 2: Dataset Preparation (COMPLETED)

**⚠️ IMPORTANT**: Proses splitting sekaligus dilakukan saat cropping.

#### Workflow Aktual (Yang Digunakan)

```
Pre-organized dataset (93k frames)
  ↓
crop_organized_dataset.py
  ↓
671,649 person crops (proper splits!)
  ↓
NO annotation needed! ✅
```

#### Statistik Dataset

```
✅ Total Crops: 671,649
✅ Train:       541,993 (80%)
✅ Val:         64,759 (10%)
✅ Test:        64,897 (10%)

<!-- Class Distribution (Balanced):
- engaged:             234k crops (34.9%)
- disengaged:          238k crops (35.4%)
- moderately-engaged:  199k crops (29.7%)
``` -->

#### Command Used:
```bash
cd phase2_dataset
python crop_organized_dataset.py
```

**Time Saved**: ~100 jam annotation manual! 🎉

---

### ✅ FASE 3: Training Classifier (COMPLETED!)

**Goal**: Train YOLO11s-cls untuk klasifikasi engagement level

#### Final Training Configuration

```bash
cd phase3_training

python train_classifier.py \
  --data "..\phase2_dataset\person_crops_organized" \
  --model yolo11s-cls.pt \
  --epochs 50 \
  --batch 160 \
  --imgsz 224 \
  --patience 10 \
  --save-period 5 \
  --workers 4 \
  --cache ram \
  --amp \
  --optimizer AdamW \
  --lr0 0.001 \
  --device 0 \
  --name engagement_organized_full \
  --exist-ok
```

#### Training Results

```
✅ Total Epochs: 50 -- Stop setelah 17 epoch karena sudah diminishing results, meskipun patience belum trigger.
✅ Training Time: ~12-15 hours
✅ Final Accuracy: 96.66%
✅ Best F1-Score: 0.9655 (macro-averaged)
✅ Model Size: YOLO11s-cls (~32MB)
```

#### Model Location

```
phase3_training/runs/engagement_organized_full/weights/best.pt
```

#### Evaluate Model

```bash
# Run detailed evaluation
python detailed_evaluation_report.py

# Output includes:
# - Overall metrics
# - Per-class performance
# - Confusion matrix
# - Misclassification analysis
# - Performance ranking
```

**Status**: ✅ Model production-ready dengan performa excellent!

---

### 🔜 FASE 4: Full Pipeline Integration (ONGOING)

**Goal**: Integrasikan detection, tracking, dan classification

#### Pipeline Flow

```
Video Input
  ↓
Person Detection (YOLOv11)
  ↓
Multi-Object Tracking (BoT-SORT)
  ↓
Engagement Classification (Trained Model)
  ↓
Visualization + Analytics
  ↓
Output: Tracked video + CSV data
```

---

## 📈 Progress Checklist

- [x] **Phase 1**: POC validation
  - [x] Person detection working
  - [x] Tracking dengan ID unik
  - [x] Basic engagement scoring
  - [x] Visualisasi hasil

- [x] **Phase 2**: Dataset preparation
  - [x] Extract frames dari video
  - [x] Crop persons otomatis (671k crops!)
  - [x] Data organization (train/val/test)
  - [x] Quality validation

- [x] **Phase 3**: Model training
  - [x] Training setup & configuration
  - [x] Model training (50 epochs)
  - [x] Hyperparameter tuning
  - [x] Evaluation & metrics (96.66% accuracy!)
  - [x] Best model selection

- [ ] **Phase 4**: Pipeline integration
- [x] Person detection (YOLOv11)
- [x] Multi-person tracking (BoT-SORT)
- [x] Engagement classification (trained model)
- [x] Test video diluar dataset OUC-CGE
- [x] Export hasil tracking
- [ ] Analytics dashboard

---

## 🔬 Technical Details

### Models Used

1. **Detection**: YOLOv11s (Ultralytics)
2. **Tracking**: Bot-SORT
3. **Classification**: YOLOv11s-cls
4. **Pose Estimation**: YOLOv11n-pose (hanya di POC penggunaannya)

### Dataset

- **Source**: OUC-CGE Engagement Dataset
- **Total Videos**: ~7696 videos
- **Total Frames**: ~93,000 frames
- **Person Crops**: 671,649 crops
- **Classes**: 3 (engaged, disengaged, moderately-engaged)

### Training Specs

- **GPU**: NVIDIA RTX 3050 Laptop GPU with 4GB VRAM (CUDA enabled)
- **Framework**: PyTorch + Ultralytics
- **Training Time**: 12-15 hours
- **Batch Size**: 160
- **Image Size**: 224x224
- **Optimizer**: AdamW
- **Learning Rate**: 0.001

---

## 📊 Key Achievements

✅ **Dataset Creation**: 671,649 person crops dengan proper splits
✅ **Time Saved**: ~100 jam annotation manual
✅ **Model Accuracy**: 96.66% overall accuracy
✅ **Balanced Performance**: All classes >95% F1-score
✅ **Production Ready**: Model siap untuk deployment

---

## 🎯 Next Steps

1. **Phase 4 Implementation**:
   - Integrate semua komponen ke pipeline
   - Optimize untuk real-time processing
   - Build analytics dashboard

2. **Testing & Validation**:
   - Test pada video baru
   - Validate tracking consistency
   - Measure end-to-end performance

3. **Documentation**:
   - API documentation
   - Deployment guide
   - User manual

---

## 📝 Research Paper Highlights

**Key Contributions**:

1. Novel person-level engagement classification approach
2. Robust to camera angles and classroom layouts
3. Individual student tracking with unique IDs
4. High accuracy (96.66%) dengan balanced performance
5. Scalable pipeline for real-world deployment

**Advantages Over Frame-Level Approach**:

- Better generalization across different classroom setups
- Individual student insights vs aggregate metrics
- More actionable data for educators
- Robust to occlusions and lighting variations

---

## 🤝 Contributing

Project ini adalah bagian dari skripsi. Untuk pertanyaan atau diskusi:

- Email: [your-email]
- GitHub Issues: [repository-link]

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- **Dataset**: OUC-CGE Engagement Dataset
- **Framework**: Ultralytics YOLO
- **Advisor**: [Advisor Name]

---

**Last Updated**: November 16, 2025
**Status**: Phase 3 Completed ✅ | Phase 4 In Planning 🔜
**Model**: YOLO11 | **Accuracy**: 96.66% 🎉
#   s t u d e n t _ e n g a n g e m e n t 
 
 #   s t u d e n t _ e n g a n g e m e n t  
 