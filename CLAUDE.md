# Classroom Engagement Analyzer — Project Context

## Project Overview
Thesis project (Skripsi) — 2-stage YOLOv11 student engagement detection & tracking system.  
Detects **2 classes: Engaged / Not-Engaged** (3-class Medium merged into Engaged per V5/V7 breakthrough).  
Full-stack: FastAPI backend + Streamlit frontend + BotSORT tracker.

## Tech Stack
| Layer | Technology |
|-------|------------|
| Detector | YOLOv11m `best_v5.pt` (2-class, mAP@50 76.2%) |
| Classifier | YOLOv11s-cls `best_v10.pt` (2-class, ROC-AUC 0.886) |
| Tracker | BotSORT (custom config, frame_rate=3, track_buffer=18) |
| Backend | FastAPI (Python), Supabase (PostgreSQL + Storage) |
| Frontend | Streamlit (multi-page) |
| Config | `.env` file, `backend/config.py` |
| Runtime | Python 3.10+, Windows, `.venv\` |

## Key File Map
```
person-tracking-engagement/
├── .env                           # Secrets + runtime config (model paths, thresholds)
├── start.ps1 / stop.ps1           # One-click start/stop (Backend + Frontend)
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Env var loader — CLASSIFY_THRESHOLD, FRAME_STRIDE, etc.
│   ├── routers/
│   │   ├── videos.py              # POST /videos/upload + job queue + save threshold/stride
│   │   ├── results.py             # GET /results/{job_id}, 2-class distribution mapping
│   │   └── auth.py                # Login/logout via Supabase auth
│   └── services/
│       ├── pipeline_service.py    # Calls TwoStagePipeline with kwargs
│       ├── analysis_service.py    # Majority vote 2-class, legacy medium→engaged mapping
│       └── supabase_service.py    # Persist: engaged_votes, not_engaged_votes per track
├── frontend/
│   ├── app.py                     # Streamlit entry point (login gate)
│   ├── fe_config.py               # ENGAGEMENT_COLORS/LABELS/EMOJI/CHART_COLORS (2-class)
│   ├── pages/
│   │   ├── 1_Upload.py            # Video upload + trigger backend job
│   │   ├── 2_Results.py           # Live-refresh job status + 2-class charts
│   │   ├── 3_History.py           # Past session history (2-class distribution dots)
│   │   └── 4_Profile.py           # User profile
│   ├── components/
│   │   ├── charts.py              # Pie/bar/stacked 2-class Plotly helpers
│   │   └── styles.py              # CSS injection helpers
│   └── services/                  # API calls to FastAPI backend
├── phase4_pipeline/
│   ├── full_pipeline.py           # TwoStagePipeline (detect best_v5 → crop → classify best_v10)
│   │                              # FullPipeline retained as backward-compat shim
│   ├── custom_botsort.yaml        # Tracker config: frame_rate=3, track_buffer=18
│   └── analyze_result.py         # Post-processing & confusion matrix
├── phase3_training/
│   ├── rangkuman_training.md      # Full training history V1–V10
│   └── calibrate_v10.py          # Per-session threshold calibration script
├── supabase_setup.sql             # DB schema — 2-class columns + classify_threshold/frame_stride
└── models/
    ├── best_v5.pt                 # Detector: YOLOv11m 2-class, mAP@50 76.2%
    └── best_v10.pt                # Classifier: YOLOv11s-cls 2-class, ROC-AUC 0.886
```

## Model Performance (Current — Deploy v5)

### Pipeline: 2-Stage Detect + Classify
- **Frame stride:** 5 (3 fps efektif dari source 15 fps — 5× speedup)
- **Threshold:** 0.170 (per-session calibration, sesi `Kelas9_2mar_0906`)

### Classifier V10 (best_v10.pt) — Test Results
| Metric | Default thr=0.5 | **Calibrated thr=0.170** |
|--------|-----------------|--------------------------|
| Top-1 Accuracy | 74.6% | **82.21%** ✓ |
| Engaged F1 | 0.767 | **0.861** |
| NotEngaged F1 | 0.722 | **0.754** |
| Macro F1 | 0.744 | **0.807** |
| ROC-AUC | **0.886** | **0.886** |

### Detector V5 (best_v5.pt)
- mAP@50: 76.2% | Engaged: 85.3% | NotEngaged: 67.0%
- Class output **diabaikan** saat inference — hanya bbox + track_id yang dipakai; label final dari classifier V10

### Training History Summary (V1–V10)
| Versi | Task | Metrik Utama | Catatan |
|-------|------|-------------|---------|
| V1–V4 | Detect 3-class | mAP ~54–58% | Plateau — bottleneck: ambiguitas Medium |
| V5 | Detect 2-class | mAP **76.2%** | Breakthrough: merge Medium→Engaged |
| V6 | Classify 3-class | Acc 57.7% | Medium tetap F1 0.335 walau crop |
| V7 | Classify 2-class | Acc **76.8%** | Crop + 2-class, baseline terbaik tanpa calibration |
| V8–V9 | Classify 2-class | Acc 73–70% | Bias NotEngaged semakin parah |
| **V10** | Classify 2-class | Acc **82.21%** (calib) | ROC-AUC 0.886, per-session calibration |

## Key Design Decisions
1. **2-Stage Pipeline** — `best_v5.pt` detect siswa → crop bbox → `best_v10.pt` classify per crop
2. **2-Class Only (Engaged/NotEngaged)** — Medium di-merge ke Engaged; ambiguitas visual Medium terbukti tidak bisa dipecahkan (ceiling 3-class ~57%)
3. **Per-Session Threshold Calibration** — threshold 0.170 (bukan default 0.5); dikalibrasi via 20% calibration set dari sesi test; dapat di-recalibrate per-konteks via `CLASSIFY_THRESHOLD` env
4. **Frame Stride 5 (3 fps efektif)** — engagement perilaku temporal orde detik, bukan milidetik; 5× speedup tanpa kehilangan akurasi
5. **Session-Based Dataset Split** — frame dari sesi rekaman yang sama tidak split antar train/val/test (no data leakage)
6. **BotSORT @3fps** — `frame_rate=3`, `track_buffer=18` (≈6 detik), `new_track_thresh=0.4` untuk cegah ID explosion
7. **Temporal Smoothing** — `EngagementSmoother` window 10 frame (≈3.3 detik @3fps), confidence-weighted majority voting
8. **Background job queue** in FastAPI — long videos don't timeout HTTP requests
9. **Supabase** for auth, storage (video uploads), and results persistence

## Common Commands (use `rtk` prefix)
```bash
# Activate venv (Windows)
.\.venv\Scripts\activate

# Start everything
.\start.ps1

# Run pipeline directly (2-stage)
python phase4_pipeline/full_pipeline.py --source <video_path>

# Per-session threshold calibration
python phase3_training/calibrate_v10.py

# Git
rtk git status
rtk git diff
rtk git log
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## Environment Variables (`.env`)
```ini
# Models
DETECTION_MODEL_PATH=models/best_v5.pt
CLASSIFIER_MODEL_PATH=models/best_v10.pt

# Pipeline config
CLASSIFY_THRESHOLD=0.170
FRAME_STRIDE=5
SMOOTHING_WINDOW=10
CLASSIFIER_IMGSZ=224
CONF_THRESHOLD=0.3

# Supabase
SUPABASE_URL=...
SUPABASE_KEY=...
BACKEND_URL=http://localhost:8000
```

## Current Status (May 2026)
- ✅ Full-stack app stable and operational (Deploy v5)
- ✅ 2-stage pipeline: detect (best_v5.pt) + classify (best_v10.pt)
- ✅ Target ≥80% accuracy TERCAPAI — V10 calibrated = **82.21%**
- ✅ Per-session threshold calibration implemented (thr=0.170)
- ✅ Frame stride 5 (3 fps efektif, 5× speedup)
- ✅ BotSORT recalibrated for 3 fps (track_buffer=18, frame_rate=3)
- ✅ Database migrated ke schema 2-class (DROP & RECREATE, disepakati di bimbingan)
- 📝 Thesis writeup in progress

## Notes for AI Assistant
- This is a **thesis project** — correctness and explainability matter more than speed
- Prefer modifying existing files over creating new ones
- Model paths set in `.env` — never hardcode them
- `full_pipeline.py` contains `TwoStagePipeline` (current) and `FullPipeline` (shim, backward-compat only)
- Classifier threshold default 0.170 (not 0.5) — changing it affects classification results significantly
- Streamlit reruns the whole script on interaction — use `st.session_state` for persistence
- DB schema is 2-class: `engaged_votes` + `not_engaged_votes` (no `moderate_votes`)
- For training context (V1–V10 history, dataset decisions): see `phase3_training/rangkuman_training.md`
- For deployment pipeline history (Deploy v1–v5): see `phase4_pipeline/rangkuman_deployment.md`
