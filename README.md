# Classroom Engagement Analyzer (YOLOv11)

> **Status Update (April 2026)**: 🔥 Fase Transisi ke Arsitektur *Single-Pass* Selesai! Model telah dilatih secara lokal menggunakan *Session-based Splitting* untuk mencegah *Data Leakage*. Aplikasi Website Terintegrasi Penuh (FastAPI + Streamlit).

Sistem deteksi cerdas untuk menganalisis dan melacak (tracking) tingkat keterlibatan siswa (*Student Engagement*) di ruang kelas secara *End-to-End* menggunakan model YOLOv11 secara lokal.

## 📋 Deskripsi Evolusi Sistem

Awalnya, sistem ini menggunakan arsitektur *Two-Pass* (Deteksi $\rightarrow$ Crop $\rightarrow$ Klasifikasi) pada dataset OUC-CGE, namun ditemukan masalah *Model Blindness* dan *Data Leakage* akibat *Random Splitting*. 

Sistem kini menggunakan pendekatan iterasi gen-2: **Single-Pass Object Detection**.
Model langsung melihat gambar kelas utuh (Full Frame) dan mendeteksi serta mengklasifikasikan siswa dalam satu tarikan napas. Pendekatan ini mengembalikan "Konteks Spasial" (Model paham murid yang menelungkup di meja atau duduk santai di atas meja) tanpa harus membuat ribuan potongan gambar kecil (*crops*).

---

## 🎯 Komparasi Pendekatan (Lama vs Baru)

| Metodologi | Fase Awal (Cropping + Cloud) | Fase Saat Ini (Single-Pass Lokal) |
|---|---|---|
| **Pemisahan Data** | *Random Frame Splitting* (Memicu Data Leakage) | **Session-based Splitting** (Anti Data Leakage) |
| **Arsitektur Model** | 2 Model (YOLO Detect + YOLO Classify) | **1 Model Tunggal** (YOLOv11s Object Detection) |
| **Konteks Lingkungan** | Buta (Sering gagal deteksi karena asums crop) | **Superior** (Paham posisi meja, postur menelungkup/tidur) |
| **Akurasi di Kertas** | 96%++ (Angka palsu akibat overfitting) | **57.8% (mAP@50)** (Angka sangat jujur & realistis) |
| **Akurasi Dunia Nyata** | Hanya melacak 3-5 subjek terdepan | **Melacak 12+ subjek stabil hingga baris belakang** |

---

## 📊 Kinerja Model Lokal Saat Ini

Model terbaru dilatih menggunakan GPU Colab (T4) pada dataset mandiri (Skripsi) berjumlah ~1117 frames dengan Augmentasi Dinamis (Mosaic 1.0, HSV, Mixup).

### Evaluasi Metrik (mAP@50)
- **Overall:** `57.8%` (Sangat stabil untuk deteksi kelas ramai)
- **High Engagement (Terlibat):** `79.4%` (Paling Stabil)
- **Low Engagement (Tidak Terlibat):** `57.2%`
- **Medium Engagement (Menengah):** `36.6%` (Area ambigu yang wajar secara teori visi komputer)
- **Laju Inferensi (Speed):** ~5.5ms per Gambar (Sangat responsif untuk edge-device)

---

## 💻 Struktur Project & Tech Stack

Aplikasi kini sepenuhnya terdistribusi (*Microservice*) untuk skalabilitas masa depan:
- **Backend Server (`backend/`)**: FastAPI + Supabase. Mengurus *job queueing* di *background* sehingga video berdurasi panjang tidak membuat antarmuka membeku (Timeout-safe).
- **Frontend App (`frontend/`)**: Streamlit UI yang dinamis dengan fitur *Live Refresh* dan integrasi sistem Login/History rekaman.
- **Data Pipeline (`phase4_pipeline/`)**: Inti pelacakan menggunakan `full_pipeline.py` (YOLOv11 + BotSORT Tracker).

---

## 🚀 Quick Start (Menjalankan Aplikasi)

### 1. Kebutuhan Instalasi
Pastikan Anda memiliki Python 3.10+ dan virtual environment aktif.
```bash
# Windows
.\venv\Scripts\activate

# Install Dependensi Utama
pip install -r requirements.txt
```

### 2. File Konfigurasi Lingkungan (`.env`)
Ubah atau pastikan variabel model pada `.env` sudah menunjuk ke model cerdas terbaru kita:
```ini
# Path ke model lokal yang telah di-_finetune_
CLASSIFIER_MODEL_PATH=models/local_best.pt

# Threshold Deteksi 
CONF_THRESHOLD=0.3
```

### 3. Menjalankan Server & UI (Satu Klik)
Kami menyiapkan script `start.ps1` untuk menyalakan Backend FastAPI dan Frontend Streamlit secara bersamaan.
```bash
./start.ps1
```
Aplikasi dapat segera diakses di browser pada: **`http://localhost:8501`**

---

## 🛣️ Rencana Pengembangan & Saran (*Future Works*)

Untuk pengerjaan jangka panjang dan pengujian Skripsi pada video asli berdurasi panjang (Misal 10 Menit / 9.000 Frames), sangat disarankan menganut implementasi **Frame Skipping**.
Mengingat perilaku psikologis manusia (ngobrol / menunduk / belajar) berjalan lambat dan persisten, deteksi tidak perlu dipaksakan secara 15 FPS. Mengubah eksekusi inferensi menjadi **3 FPS** akan memangkas beban GPU lokal hingga 5x Lipat tanpa merusak akurasi agregat analisis sistem.

---
**Status Terakhir**: Aplikasi Stabil & Model Lokal Operasional. 🎓🏁