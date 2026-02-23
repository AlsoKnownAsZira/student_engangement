# ═══════════════════════════════════════════════════════
# Stop all Engagement Analyzer services
# ═══════════════════════════════════════════════════════

function Free-Port($port) {
    $procIds = netstat -ano | Select-String "LISTENING" |
        Select-String ":$port " |
        ForEach-Object { ($_ -split '\s+')[-1] } |
        Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
    foreach ($p in $procIds) {
        try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {}
    }
}

Write-Host "Stopping services..." -ForegroundColor Yellow
Free-Port 8000
Free-Port 8501
Start-Sleep 1
Write-Host "All services stopped." -ForegroundColor Green
