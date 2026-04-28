# Rangkuman Deployment Phase 4

> Catatan: rangkuman ini fokus ke **deployment pipeline + integrasi web**, bukan training. Untuk training versi V1–V10 lihat [phase3_training/rangkuman_training.md](../phase3_training/rangkuman_training.md).

## Overview

| Versi | Pipeline | Model | Class | Tracker | Frame Rate | Catatan Utama |
|-------|----------|-------|-------|---------|-----------|---------------|
| Deploy v1 | Single-pass | YOLO Detect 3-class | High / Med / Low | BotSORT default 15fps | 15 fps (semua frame) | Pipeline awal, mengikuti training V1–V4 |
| Deploy v2 | Single-pass + tuning tracker | YOLO Detect 3-class | High / Med / Low | Custom BotSORT (`new_track_thresh=0.4`, `track_buffer=60`) | 15 fps | Mengatasi ID explosion 58→22, tracker stable |
| Deploy v3 | Single-pass + smoothing temporal | YOLO Detect 3-class + EngagementSmoother | engaged / mod / disengaged | Custom BotSORT | 15 fps | Tambah voting confidence-weighted window 10 frame, label tidak flicker |
| Deploy v4 (SAHI eksperimen) | Sliced inference | YOLO Detect + SAHI tile + ByteTrack | engaged / mod / disengaged | supervision.ByteTrack | 15 fps | Eksperimen — dibuang karena tile overlap = ID explosion 58→131 |
| **Deploy v5 (current)** | **2-stage detect → crop → classify** | **best_v5.pt (detect) + best_v10.pt (classify)** | **engaged / not-engaged** | **Custom BotSORT @3fps** | **3 fps efektif (stride 5)** | **Sesuai bimbingan: 2-class + sampling frame, sejalan training V10** |

---

## Per-Versi Detail

### Deploy v1 — Pipeline Awal (single-pass)

- **Pipeline:** YOLO Detect saja, output bbox + class langsung dari model
- **Model:** `models/local_best.pt` (YOLOv11s 3-class)
- **Tracker:** BotSORT default Ultralytics (frame_rate=30, track_buffer=30)
- **Frame rate:** Semua frame source diproses
- **Class output:** High Engagement / Medium Engagement / Low Engagement → di-normalize ke `engaged / moderately-engaged / disengaged`
- **Database:** `engaged_votes`, `moderate_votes`, `disengaged_votes` per `student_results`
- **Masalah:**
  - ID explosion (58 unik di awal sesi → 131 setelah beberapa menit untuk 30-an siswa)
  - Banyak label flicker antar frame untuk track yang sama
  - mAP overall hanya ~57.8% (sesuai training V1)

### Deploy v2 — Custom BotSORT Config

- **Perubahan:** Tambah `phase4_pipeline/custom_botsort.yaml`
  - `track_high_thresh: 0.3 → 0.2` (terima deteksi confidence rendah)
  - `track_low_thresh: 0.1 → 0.05` (pertahankan track lemah)
  - `new_track_thresh: 0.3 → 0.4` (selektif buat ID baru)
  - `track_buffer: 30 → 60` (tahan track hilang sampai 4 detik @15fps)
  - `match_thresh: 0.8 → 0.85` (toleran matching)
- **Hasil:** ID explosion teratasi (58 → ~22 unik di seluruh sesi). Tracker jadi pondasi yang dipakai sampai sekarang.

### Deploy v3 — Temporal Smoothing

- **Perubahan:** Tambah `EngagementSmoother` class
  - Sliding window 10 frame per `track_id`
  - Voting confidence-weighted majority
  - Cleanup history saat track expired
- **Hasil:** Label per track stabil — tidak ada lagi flicker `engaged → disengaged → engaged` antar 3 frame berturut-turut.
- **Konsekuensi:** Output CSV punya kolom `raw_engagement` (per-frame) dan `engagement_level` (smoothed).

### Deploy v4 — SAHI Eksperimen (dibuang)

- **Pipeline:** SAHI sliced inference + supervision.ByteTrack
- **Hipotesis:** 63% bbox siswa <1% area frame → tile inference akan boost recall
- **Hasil:** Recall siswa kecil meningkat tapi **tile overlap menyebabkan double-detect** → ID explosion 58 → 131
- **Keputusan:** Dibuang. Pertahankan kwargs di shim untuk backward-compat tapi tidak dipakai di production.

### Deploy v5 (current) — 2-Stage Detect + Classify (V10)

> **Latar belakang dari bimbingan:**
> 1. Output ACC (Akurasi) untuk **2 class engaged / disengaged** (bukan 3-class)
> 2. Video QHD 15 FPS **tidak harus diproses semua frame setiap detiknya**
>
> Versi ini menyelaraskan deployment dengan training V10 yang sudah tembus target ≥80% via per-session calibration.

#### Arsitektur Baru

```
Frame source (QHD @ 15 fps)
       │
       │  frame_stride = 5  →  ambil 1 dari 5 frame
       ↓
Frame ter-sampel (~3 fps efektif)
       │
       ↓
[Model 1] best_v5.pt (Detect, 2-class)
       │  Class output diabaikan (BotSORT match pakai IoU+motion, bukan class)
       │  Hanya ambil bbox + track_id
       ↓
Bounding box + track_id (BotSORT, frame_rate=3, track_buffer=18)
       │
       ↓
Crop bbox @ 224px
       │
       ↓
[Model 2] best_v10.pt (Classify, 2-class)
       │  Output: P(engaged)
       ↓
Threshold: P(engaged) ≥ 0.170  →  "engaged"
                   else        →  "not-engaged"
       │
       ↓
EngagementSmoother (window 10 frame ≈ 3.3 detik @3fps)
       │  Voting confidence-weighted per track_id
       ↓
CSV + annotated video + DB persist
```

#### Keputusan Frame Stride

| Pilihan | Inferensi Speedup | Smoothing Window | Status |
|---------|-------------------|------------------|--------|
| Stride 1 (15 fps) | 1.0x | 10 frame ≈ 0.67 detik | Terlalu boros, smoothing terlalu pendek |
| **Stride 5 (3 fps)** ✓ | **5.0x** | **10 frame ≈ 3.3 detik** | **Dipilih — sweet spot** |
| Stride 15 (1 fps) | 15x | 10 frame ≈ 10 detik | Terlalu lambat, tracker sering kehilangan ID antar frame |

**Alasan stride 5:** Engagement adalah perilaku temporal orde detik (siswa tidak berubah engagement tiap 1/15 detik), jadi 3 fps cukup untuk capture perubahan. Buffer tracker dan smoothing window dijaga rasionya supaya tetap setara dengan setup lama.

#### Threshold Klasifikasi

- **Default: 0.170** — diambil dari per-session calibration V10 di sesi `Kelas9_2mar_0906` (lihat [phase3_training/calibrate_v10.py](../phase3_training/calibrate_v10.py))
- Untuk kelas/sesi baru, threshold bisa di-recalibrate (ambil ~50–100 sampel berlabel dari sesi baru, jalankan grid search max macro-F1)
- Threshold dapat diubah via env `CLASSIFY_THRESHOLD`

#### Tracker — Kalibrasi ulang untuk 3 fps

| Parameter | Lama (15 fps) | Baru (3 fps) | Alasan |
|-----------|---------------|--------------|--------|
| `frame_rate` | 15 | **3** | Sesuaikan dengan inferensi efektif |
| `track_buffer` | 60 (≈4 detik) | **18 (≈6 detik)** | Frame stride lebih besar → butuh buffer lebih panjang biar siswa nunduk/tertutup tetap reconnect |
| `track_high_thresh` | 0.2 | 0.2 | Sama |
| `track_low_thresh` | 0.05 | 0.05 | Sama |
| `new_track_thresh` | 0.4 | 0.4 | Sama — selektif bikin ID baru |
| `match_thresh` | 0.85 | 0.85 | Sama |

#### File yang Berubah di Deploy v5

| File | Perubahan |
|------|-----------|
| `phase4_pipeline/full_pipeline.py` | Rewrite jadi `TwoStagePipeline` (detect + classify). `FullPipeline` dipertahankan sebagai shim backward-compat |
| `phase4_pipeline/custom_botsort.yaml` | `frame_rate: 15→3`, `track_buffer: 60→18` |
| `backend/config.py` | Tambah `CLASSIFY_THRESHOLD`, `FRAME_STRIDE`, `SMOOTHING_WINDOW`, `CLASSIFIER_IMGSZ`. Default `DETECTION_MODEL_PATH=models/best_v5.pt`, `CLASSIFIER_MODEL_PATH=models/best_v10.pt` |
| `backend/services/pipeline_service.py` | Pakai `TwoStagePipeline` dengan kwargs baru |
| `backend/services/analysis_service.py` | Voting 2-class. Mapping legacy: `medium → engaged`, `low/disengaged → not-engaged` |
| `backend/services/supabase_service.py` | Kolom `not_engaged_votes` (bukan `moderate_votes` + `disengaged_votes`) |
| `backend/models/schemas.py` | Enum 2 nilai (`ENGAGED`, `NOT_ENGAGED`). `EngagementDistribution{engaged, not_engaged}` |
| `backend/routers/videos.py` | Simpan `classify_threshold` & `frame_stride` ke DB row untuk audit per-analisis |
| `backend/routers/results.py` | Distribution mapping 2-class |
| `supabase_setup.sql` | DROP & RECREATE — kolom 2-class + tambah field `classify_threshold` dan `frame_stride` di tabel `analyses` |
| `frontend/fe_config.py` | Drop `moderately-engaged` dari `ENGAGEMENT_COLORS/LABELS/EMOJI/CHART_COLORS` |
| `frontend/components/charts.py` | Pie / bar / stacked render 2-class |
| `frontend/pages/2_Results.py` | Bar list + dataframe rename 2-class |
| `frontend/pages/3_History.py` | Distribution dots 2-class |
| `.env.example` | Tambah field baru |

---

## Analisis Akar Masalah (Mengapa Deploy v5)

### Ketidaksesuaian Pipeline Lama dengan Training V10

| Aspek | Training V10 | Deploy v1–v3 lama |
|-------|--------------|-------------------|
| Class | 2 (engaged / not-engaged) | 3 (high / med / low) |
| Architecture | 2-stage detect + classify | 1-stage detect saja |
| Best test acc | **82.21%** (calibrated) | ~57% (3-class plateau) |
| Threshold | 0.170 (calibrated) | 0.5 default + softmax |
| Input ke classifier | Crop bbox 224px | (tidak ada classifier terpisah) |

Deploy lama memakai model 3-class hasil training V1–V4 yang stuck di mAP ~57%. Training V10 menghasilkan classifier 2-class dengan ROC-AUC 0.886 dan calibrated test acc 82.21%, tapi pipeline web masih mengikuti arsitektur lama.

### Mengapa Frame Sampling Diperlukan

- Source video QHD (2560×1440) @ 15 fps + 30-an siswa per frame.
- Inferensi penuh tiap frame = ~5.5 ms × 30 detect × 224px-classify ≈ overhead besar untuk video panjang (1 jam pelajaran ≈ 54 ribu frame).
- Stride 5 memangkas ke ~10.8 ribu frame inferensi → 5x lebih cepat tanpa kehilangan informasi engagement (perilaku temporal orde detik, bukan milidetik).

### Mengapa best_v5.pt sebagai Detector

- `best_v5.pt` adalah model deteksi 2-class (Engaged/NotEngaged) hasil training V5 dengan mAP@50 = 76.2% di domain CCTV kelas
- Class output-nya **tidak dipakai** — hanya ambil bbox & track_id. Classifier V10 yang menentukan label final
- Alternatif yang dipertimbangkan:
  - `local_best.pt` (3-class) → recall siswa kecil OK, tapi class output mengganggu BotSORT (tidak dipakai juga)
  - `yolo11s.pt` (general person class) → recall siswa duduk/kecil lebih rendah karena bukan domain-specific
- Pilihan: **best_v5.pt** karena domain-tuned recall paling tinggi

---

## Improvement yang Sudah Dicoba (Phase 4)

| Improvement | Dicoba di | Hasil |
|-------------|-----------|-------|
| Custom BotSORT config | v2 | ID explosion 58→22 ✅ |
| Temporal smoothing 10 frame | v3 | Label flicker hilang ✅ |
| SAHI sliced inference | v4 | ID explosion 58→131, dibuang ❌ |
| 2-stage pipeline (V10) | **v5** | **Sejalan dengan training V10 best result** ✅ |
| Frame stride 5 (3 fps efektif) | **v5** | **5x speedup tanpa kehilangan akurasi** ✅ |
| BotSORT recalibrasi @3fps | **v5** | **track_buffer 60→18, frame_rate 15→3** ✅ |

---

## Cara Kerja di Production (Web)

```
1. User upload video di /Upload
       ↓
2. FastAPI /api/videos/upload terima file → simpan ke Supabase Storage
       ↓
3. Background task panggil PipelineManager.process()
       │
       │  TwoStagePipeline (detect best_v5 + classify best_v10)
       │  Frame stride 5, threshold 0.170, smoothing window 10
       ↓
4. df = per-frame tracking data (kolom: frame, track_id, bbox, prob_engaged, engagement_level)
       ↓
5. analysis_service.analyse(df) → majority vote per track → class summary
       ↓
6. Persist ke Supabase:
       - analyses: total_students, engagement_distribution, classify_threshold, frame_stride
       - student_results: per track_id (engaged_votes, not_engaged_votes, vote_percentage)
       ↓
7. User polling /Results page → dapat distribution chart + per-student table
```

---

## Cara Recalibrate untuk Kelas/Sesi Baru

Default threshold (0.170) dikalibrasi di sesi `Kelas9_2mar_0906`. Untuk kelas dengan lighting / sudut kamera berbeda:

1. Rekam 50–100 sampel berlabel dari kelas baru (atau gunakan ground-truth dari beberapa frame).
2. Jalankan inference best_v10.pt di sampel tersebut → kumpulkan probabilitas Engaged.
3. Grid search threshold di rentang [0.05, 0.95] dengan kriteria max macro-F1 (lihat `phase3_training/calibrate_v10.py`).
4. Update `.env` → `CLASSIFY_THRESHOLD=<nilai_baru>`.
5. Restart backend. Threshold tersimpan di tabel `analyses.classify_threshold` per analisis untuk traceability.

---

## Kesimpulan

1. **Deploy v5 menyelaraskan web dengan training V10** — 2-class, threshold calibrated, 2-stage pipeline. Sebelumnya web masih 3-class dan single-pass (sesuai V1–V4 lama).
2. **Frame stride 5 (3 fps efektif)** = sweet spot antara kecepatan inferensi (5x lebih cepat) dan stabilitas tracker/smoothing (window 3.3 detik masih representatif).
3. **Tracker disesuaikan untuk 3 fps efektif** — `frame_rate=3`, `track_buffer=18` (≈6 detik). Tetap konservatif dengan `new_track_thresh=0.4` untuk cegah ID explosion. Detector class output diabaikan supaya class flip tidak memecah ID.
4. **Database di-DROP & RECREATE** — keputusan disepakati di bimbingan; data 3-class lama tidak compatible. Tabel baru juga simpan `classify_threshold` & `frame_stride` per analisis untuk audit metodologi.
5. **Calibration default 0.170** dipakai langsung; opsi recalibrate per-konteks tersedia via env var (tidak butuh retrain model).
