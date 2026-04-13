# Classroom Engagement Analyzer — Project Context

## Project Overview
Thesis project (Skripsi) — Single-Pass YOLOv11s student engagement detection & tracking system.  
Detects 3 classes: **High Engagement**, **Medium Engagement**, **Low Engagement**.  
Full-stack: FastAPI backend + Streamlit frontend + BotSORT tracker.

## Tech Stack
| Layer | Technology |
|-------|------------|
| Inference | YOLOv11s (Ultralytics) + BotSORT tracker |
| Backend | FastAPI (Python), Supabase (PostgreSQL + Storage) |
| Frontend | Streamlit (multi-page) |
| Config | `.env` file, `config.py` at root |
| Runtime | Python 3.10+, Windows, `.venv\` |

## Key File Map
```
person-tracking-engagement/
├── config.py                  # Root config loader (env vars, paths)
├── .env                       # Secrets: Supabase URL/key, model path, thresholds
├── start.ps1 / stop.ps1       # One-click start/stop (Backend + Frontend)
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── routers/
│   │   ├── videos.py          # POST /videos/upload + job queue logic
│   │   ├── results.py         # GET /results/{job_id}
│   │   └── auth.py            # Login/logout via Supabase auth
│   └── services/              # Business logic (processing, db)
├── frontend/
│   ├── app.py                 # Streamlit entry point (login gate)
│   ├── pages/
│   │   ├── 1_Upload.py        # Video upload + trigger backend job
│   │   ├── 2_Results.py       # Live-refresh job status + charts
│   │   ├── 3_History.py       # Past session history
│   │   └── 4_Profile.py       # User profile
│   ├── components/
│   │   ├── charts.py          # Plotly chart helpers
│   │   └── styles.py          # CSS injection helpers
│   └── services/              # API calls to FastAPI backend
├── phase4_pipeline/
│   ├── full_pipeline.py       # Core: YOLO inference + BotSORT + CSV output
│   ├── custom_botsort.yaml    # Tracker config (tuned)
│   └── analyze_result.py      # Post-processing & confusion matrix
└── models/
    └── local_best.pt          # Fine-tuned YOLOv11s (primary model)
```

## Model Performance (Current)
- **mAP@50 Overall:** 57.8%
- High Engagement: 79.4% | Medium: 36.6% | Low: 57.2%
- Inference speed: ~5.5ms/frame
- Training: ~1117 frames, session-based split (no data leakage)

## Key Design Decisions
1. **Single-Pass Detection** — one model, full frame (not crop + classify)
2. **Session-Based Dataset Split** — frames from same recording session never split across train/val/test
3. **BotSORT tracking** with temporal label smoothing for stable ID assignment
4. **Background job queue** in FastAPI — long videos don't timeout HTTP requests
5. **Supabase** for auth, storage (video uploads), and results persistence

## Common Commands (use `rtk` prefix)
```bash
# Activate venv (Windows)
.\.venv\Scripts\activate

# Start everything
.\start.ps1

# Run pipeline directly
python phase4_pipeline/full_pipeline.py --source <video_path> --model models/local_best.pt

# Git
rtk git status
rtk git diff
rtk git log
rtk git add . && rtk git commit -m "msg" && rtk git push

# Docker (if using containers)
rtk docker ps
rtk docker logs <container>
```

## Environment Variables (`.env`)
```ini
CLASSIFIER_MODEL_PATH=models/local_best.pt
CONF_THRESHOLD=0.3
SUPABASE_URL=...
SUPABASE_KEY=...
BACKEND_URL=http://localhost:8000
```

## Current Status (April 2026)
- ✅ Full-stack app stable and operational
- ✅ Model trained with session-based splitting
- ✅ BotSORT tracker integrated with temporal smoothing
- 🔬 Exploring performance ceiling: mAP plateau ~57-58%, evaluating SAHI & higher-res training (1280px)
- 📝 Thesis writeup in progress

## Notes for AI Assistant
- This is a **thesis project** — correctness and explainability matter more than speed
- Prefer modifying existing files over creating new ones
- The model path is set in `.env` — never hardcode it
- `full_pipeline.py` is the core inference engine; changes there need care
- Streamlit reruns the whole script on interaction — use `st.session_state` for persistence
