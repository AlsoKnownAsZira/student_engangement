# ═══════════════════════════════════════════════════════════════════
# Classroom Engagement Analyzer — Startup Script
# ═══════════════════════════════════════════════════════════════════
#
# USAGE (pick one):
#   1. Right-click this file → "Run with PowerShell"
#   2. In terminal:  powershell -ExecutionPolicy Bypass -File start.ps1
#
# Two new windows will open (backend + frontend).
# Close those windows to stop the services.
# Or run .\stop.ps1 to stop everything at once.
# ═══════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# ── Paths (edit VENV_DIR if your venv is elsewhere) ──────────────
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR    = "D:\kuliah\Skripsi\venv-ouc-cge"
$PYTHON_EXE  = "$VENV_DIR\Scripts\python.exe"

Set-Location $PROJECT_DIR

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Classroom Engagement Analyzer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Pre-flight checks ────────────────────────────────────────────
$ok = $true
foreach ($f in @($PYTHON_EXE, ".env", "models\yolo11s.pt", "models\best.pt")) {
    if (-not (Test-Path $f)) {
        Write-Host "  MISSING: $f" -ForegroundColor Red
        $ok = $false
    }
}
if (-not $ok) { Write-Host "Fix the above and try again." -ForegroundColor Red; pause; exit 1 }
Write-Host "[OK] All required files found" -ForegroundColor Green

# ── Free up ports ────────────────────────────────────────────────
function Free-Port($port) {
    $procIds = netstat -ano | Select-String "LISTENING" |
        Select-String ":$port " |
        ForEach-Object { ($_ -split '\s+')[-1] } |
        Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
    foreach ($p in $procIds) {
        try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {}
    }
}
Free-Port 8000; Free-Port 8501
Start-Sleep 1
Write-Host "[OK] Ports 8000 & 8501 are free" -ForegroundColor Green

# ── Launch backend (separate window) ─────────────────────────────
Write-Host "[..] Starting backend (loading ML models ~5-15 s)..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON_EXE `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory $PROJECT_DIR

# Poll until backend is healthy
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep 1
    $code = & $PYTHON_EXE "$PROJECT_DIR\scripts\healthcheck.py" 2>$null
    if ("$code".Trim() -eq "200") { $ready = $true; break }
    if ($i % 5 -eq 0) { Write-Host "     still loading... ($i s)" -ForegroundColor Gray }
}
if (-not $ready) {
    Write-Host "ERROR: Backend did not start in 60 s - check its window." -ForegroundColor Red
    pause; exit 1
}
Write-Host "[OK] Backend ready  ->  http://localhost:8000" -ForegroundColor Green

# ── Launch frontend (separate window) ────────────────────────────
Write-Host "[..] Starting frontend..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON_EXE `
    -ArgumentList "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.headless", "true" `
    -WorkingDirectory $PROJECT_DIR

Start-Sleep 3
Start-Process "http://localhost:8501"   # open browser

Write-Host "[OK] Frontend ready ->  http://localhost:8501  (opened in browser)" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ALL SERVICES RUNNING" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend : http://localhost:8501" -ForegroundColor White
Write-Host "  Backend  : http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  To stop: close the two server windows, or run stop.ps1" -ForegroundColor Gray
Write-Host ""
