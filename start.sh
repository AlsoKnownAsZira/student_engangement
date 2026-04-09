#!/bin/bash

# =======================================================================
# Classroom Engagement Analyzer — Startup Script (Linux)
# =======================================================================
#
# USAGE:
#   1. Open a terminal in this folder
#   2. Run: bash start.sh
#
# To stop:
#   Press Ctrl+C or run: bash stop.sh
# =======================================================================

# ── Paths (edit PYTHON_EXE if your venv is elsewhere) ────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Since Windows venvs don't work on Linux, you must use your Linux python or venv.
# If you created a virtual environment on Linux, point it here:
# PYTHON_EXE="../../venv-ouc-cge-linux/bin/python"
# Otherwise, we will default to `python3`, which uses whatever python environment is currently active in the shell.
PYTHON_EXE="python3"

echo ""
echo -e "\e[36m========================================\e[0m"
echo -e "\e[36m  Classroom Engagement Analyzer\e[0m"
echo -e "\e[36m========================================\e[0m"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────
OK=true
for f in ".env" "models/yolo11s.pt" "models/best.pt"; do
    if [ ! -f "$f" ]; then
        echo -e "\e[31m  MISSING: $f\e[0m"
        OK=false
    fi
done

if ! command -v $PYTHON_EXE &> /dev/null; then
    echo -e "\e[31m  MISSING: $PYTHON_EXE not found. Please activate your Linux venv or install python3.\e[0m"
    OK=false
fi

if [ "$OK" = false ]; then
    echo -e "\e[31mFix the above and try again.\e[0m"
    read -p "Press enter to exit..."
    exit 1
fi
echo -e "\e[32m[OK] All required files found\e[0m"

# ── Free up ports ────────────────────────────────────────────────
function free_port() {
    local port=$1
    if fuser -s "$port/tcp" 2>/dev/null; then
        fuser -k -9 "$port/tcp" 2>/dev/null
    fi
}
free_port 8000
free_port 8501
sleep 1
echo -e "\e[32m[OK] Ports 8000 & 8501 are free\e[0m"

# ── Launch backend (background) ──────────────────────────────────
echo -e "\e[33m[..] Starting backend (loading ML models ~5-15 s)...\e[0m"
$PYTHON_EXE -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Poll until backend is healthy
READY=false
for i in {1..60}; do
    sleep 1
    # Check using healthcheck.py (stripping whitespace)
    CODE=$($PYTHON_EXE "$PROJECT_DIR/scripts/healthcheck.py" 2>/dev/null | xargs)
    if [ "$CODE" == "200" ]; then
        READY=true
        break
    fi
    if [ $((i % 5)) -eq 0 ]; then
        echo "     still loading... ($i s)"
    fi
done

if [ "$READY" = false ]; then
    echo -e "\e[31mERROR: Backend did not start in 60 s.\e[0m"
    kill $BACKEND_PID 2>/dev/null
    read -p "Press enter to exit..."
    exit 1
fi
echo -e "\e[32m[OK] Backend ready  ->  http://localhost:8000\e[0m"

# ── Launch frontend (background) ─────────────────────────────────
echo -e "\e[33m[..] Starting frontend...\e[0m"
$PYTHON_EXE -m streamlit run frontend/app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!

sleep 3
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8501" &> /dev/null
fi

echo -e "\e[32m[OK] Frontend ready ->  http://localhost:8501  (opened in browser)\e[0m"
echo ""
echo -e "\e[32m========================================\e[0m"
echo -e "\e[32m  ALL SERVICES RUNNING\e[0m"
echo -e "\e[32m========================================\e[0m"
echo ""
echo "  Frontend : http://localhost:8501"
echo "  Backend  : http://localhost:8000"
echo "  API Docs : http://localhost:8000/docs"
echo ""
echo -e "\e[90m  To stop: Press Ctrl+C in this terminal, or run bash stop.sh from another terminal\e[0m"
echo ""

# Trap Ctrl+C (SIGINT) and kill background processes
trap "echo -e '\nStopping services...'; free_port 8000; free_port 8501; exit" INT

# Wait for background processes
wait
