# Keeps the paper engine running.
#
# The engine only evaluates candles while the process is alive, so a laptop
# reboot or a crashed process silently costs setups. This wrapper restarts
# uvicorn after any exit and is registered as a logon scheduled task by
# install_service.ps1.
#
# Reload is deliberately off: an editor save must not drop the scan loop.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir 'engine.log'

Set-Location $root
while ($true) {
    # A backend already started by hand owns the port. Wait for it rather than
    # spawning a second engine that cannot bind: two engines would write to one
    # journal, and the loser would restart every few seconds.
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
        continue
    }
    "$(Get-Date -Format o) starting paper engine" | Add-Content $log
    try {
        & $python -m uvicorn app.main:app --port 8000 --host 127.0.0.1 *>> $log
        "$(Get-Date -Format o) engine exited with code $LASTEXITCODE" | Add-Content $log
    } catch {
        "$(Get-Date -Format o) engine failed to start: $_" | Add-Content $log
    }
    # A failing start (bad key, port in use) would otherwise spin the CPU.
    Start-Sleep -Seconds 15
}
