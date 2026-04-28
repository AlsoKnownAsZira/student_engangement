# Classroom Engagement Analyzer (YOLOv11)

> **Status Update (April 2026)**: 🏆 Model Final Tercapai! **V10 + Per-Session Calibration** berhasil menembus target ≥80% dengan akurasi test **82.21%** dan ROC-AUC **0.886** (terbaik dari semua versi). Aplikasi Website Terintegrasi Penuh (FastAPI + Streamlit).

Sistem deteksi cerdas untuk menganalisis dan melacak (tracking) tingkat keterlibatan siswa (*Student Engagement*) di ruang kelas secara *End-to-End* menggunakan model YOLOv11 secara lokal.

---

## 📋 Deskripsi Evolusi Sistem

Sistem ini melalui 10 iterasi pelatihan model dengan dua pendekatan arsitektur utama:

- **Fase Awal (V1–V5):** *Single-Pass Object Detection* — satu model YOLOv11 mendeteksi sekaligus mengklasifikasikan siswa dari full frame.
- **Fase Lanjut (V6–V10):** *Two-Pass Pipeline* — model deteksi orang terpisah, lalu crop per individu diklasifikasikan oleh YOLOv11s-cls. Pendekatan ini memungkinkan balancing dataset per kelas secara presisi.

**Temuan kunci yang membentuk evolusi ini:**
- 3-class (High/Medium/Low) memiliki ceiling ~57–58% yang tidak bisa ditembus — ambiguitas visual kelas Medium terlalu besar.
- Merge ke **2-class** (Engaged / NotEngaged) adalah satu-satunya breakthrough, lompat dari ~57% ke **76%+**.
- Per-session calibration (V10) berhasil menembus target ≥80%.

---

## 🎯 Ringkasan Iterasi Training (V1–V10)

| Versi | Task | Model | Dataset | Metrik Test | Catatan |
|-------|------|-------|---------|-------------|---------|
| V1 | Detect 3-class | YOLOv11s | 742 img | mAP@50 57.8% | Baseline |
| V2 | Detect 3-class | YOLOv11m | 742 img | mAP@50 58.5% | +Resolusi 960px |
| V3 | Detect 3-class | YOLOv11m | 742 img | mAP@50 ~58% | 1280px, tidak membantu |
| V4 | Detect 3-class | YOLOv11m | 1818 img | mAP@50 54.8% | Augment frame offline, turun |
| V5 | Detect 2-class | YOLOv11m | 742 img | mAP@50 **76.2%** | Merge Medium→Engaged, breakthrough |
| V6 | Classify 3-class | YOLOv11s-cls | crops 30k | Top-1 Acc 57.7% | Crop per orang, Medium tetap buruk |
| V7 | Classify 2-class | YOLOv11s-cls | crops 16.7k | Top-1 Acc **76.8%** | Crop + 2-class, baseline terbaik |
| V8 | Classify 2-class | YOLOv11s-cls | crops 16.8k | Top-1 Acc 73.0% | Dataset baru, turun dari V7 |
| V9 | Classify 2-class | YOLOv11s-cls | crops balanced | Top-1 Acc 70.6% | Balanced val + AdamW, turun lagi |
| **V10** | Classify 2-class | YOLOv11s-cls | crops multi-session | **82.21%** (calibrated) | **✅ Target ≥80% TERCAPAI** |

> Untuk detail lengkap setiap versi, lihat [`phase3_training/rangkuman_training.md`](phase3_training/rangkuman_training.md).

---

## 📊 Kinerja Model Final (V10 + Per-Session Calibration)

Model akhir: **YOLOv11s-cls** dilatih pada `crops_v10` (multi-session split, tanpa offline augmentation) dengan threshold dikalibrasi per-sesi menggunakan 20% calibration set.

### Metrik Final (Final Test Set — 80% dari test set = 3390 sampel)

| Metrik | V7 (Fallback) | **V10 Calibrated (Final)** | Δ |
|--------|--------------|---------------------------|---|
| **Top-1 Accuracy** | 76.81% | **82.21%** | **+5.40%** ✅ |
| Engaged — F1 | 0.802 | **0.861** | +5.9% |
| NotEngaged — F1 | 0.721 | **0.754** | +3.3% |
| Macro F1 | 0.761 | **0.807** | +4.6% |
| Balanced Accuracy | 77.21% | **80.20%** | +3.0% |
| **ROC-AUC** | 0.862 | **0.886** | +2.4% |
| Engaged Recall | 75.5% | **88.6%** | **+13.1%** |

> **Threshold kalibrasi: 0.170** (dipilih di calibration set 20%, bukan di final test set — etis untuk dilaporkan di skripsi).

### Arsitektur Pipeline Final

```
Frame Kelas (Video)
      ↓
[Model 1] YOLO Detect (deteksi posisi setiap siswa)
      ↓
  Bounding Box tiap siswa
      ↓
  Crop bbox dari frame
      ↓
[Model 2] YOLOv11s-cls (models/best_v10.pt)
      ↓
  Probabilitas Engaged (0–1)
      ↓
  Threshold per-sesi (default: 0.170)
      ↓
  Output: Engaged / NotEngaged
```

---

## 💻 Struktur Project & Tech Stack

```
person-tracking-engagement/
├── backend/            # FastAPI — job queueing, Supabase integration
├── frontend/           # Streamlit UI — Live Refresh, Login, History
├── phase3_training/    # Script training V1–V10 + rangkuman_training.md
├── phase4_pipeline/    # Core tracking: full_pipeline.py (YOLOv11 + BotSORT)
├── models/             # Model weights (.pt)
└── requirements.txt
```

- **Backend (`backend/`)**: FastAPI + Supabase. Job queueing di background — video panjang tidak membuat UI timeout.
- **Frontend (`frontend/`)**: Streamlit dengan Live Refresh, Login, dan History rekaman.
- **Pipeline (`phase4_pipeline/`)**: `full_pipeline.py` — YOLOv11 Detect + BotSORT Tracker + YOLOv11s-cls Classifier.

---

## 🚀 Quick Start (Menjalankan Aplikasi)

### 1. Kebutuhan Instalasi

Pastikan Python 3.10+ dan virtual environment aktif.

```bash
# Aktivasi virtual environment
source venv/bin/activate         # Linux/macOS
# atau
.\venv\Scripts\activate          # Windows

# Install dependensi
pip install -r requirements.txt
```

### 2. File Konfigurasi (`.env`)

```ini
# Path ke model classifier V10
CLASSIFIER_MODEL_PATH=models/best_v10.pt

# Threshold default (dikalibrasi dari sesi 2mar_0906)
CONF_THRESHOLD=0.170
```

### 3. Menjalankan Aplikasi

```bash
# Linux
./start.sh

# Windows
./start.ps1
```

Akses di browser: **`http://localhost:8501`**

---

## 🔬 Keputusan Model & Fallback

| Skenario | Model yang Digunakan |
|----------|----------------------|
| **Production (default)** | V10 + per-session calibration — akurasi terbaik (82.21%) |
| Tanpa calibration data | V7 (crop + 2-class classify) — 76.81%, pipeline sama |
| Pipeline sederhana 1-model | V5 (detect langsung) — mAP 76.2%, tanpa crop step |

> Per-session calibration: saat digunakan di sesi/kelas baru, sediakan 50–100 sampel berlabel dari sesi tersebut untuk mencari threshold optimal sebelum inference massal.

---

## 🛣️ Rencana Pengembangan (*Future Works*)

- **Frame Skipping (3 FPS)**: Perilaku engagement berjalan lambat — inferensi 15 FPS boros tanpa manfaat akurasi. Menurunkan ke 3 FPS memangkas beban GPU ~5x.
- **Multi-class Temporal Modeling**: Gunakan urutan frame (LSTM/Transformer) untuk memodelkan engagement berbasis waktu, bukan per-frame.
- **Auto-Calibration UI**: Tambahkan fitur upload sampel berlabel di frontend untuk kalibrasi threshold otomatis per sesi baru.

---

**Status Terakhir**: Model Final (V10 Calibrated) Operasional. Target Skripsi ≥80% Tercapai. 🎓🏁