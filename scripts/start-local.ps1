# Starts the native Windows stack with ports taken from the root .env, so the
# API port and the Vite proxy target can never drift apart.
#   powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Virtual environment missing: create it with 'python -m venv .venv' and install backend/requirements-dev.txt" }

$envFile = @{}
if (Test-Path (Join-Path $root '.env')) {
    foreach ($line in Get-Content (Join-Path $root '.env')) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $envFile[$Matches[1]] = $Matches[2] }
    }
}
$port = if ($envFile['API_PORT']) { $envFile['API_PORT'] } else { '8000' }
$backend = if ($envFile['QUEUE_BACKEND']) { $envFile['QUEUE_BACKEND'] } else { 'redis' }

$busy = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($busy) { throw "Port $port (API_PORT) is already in use by process $($busy[0].OwningProcess). Stop it or change API_PORT in .env." }

# Keep the proxy target aligned with the port uvicorn actually binds.
$env:API_PROXY_TARGET = "http://127.0.0.1:$port"
Write-Host "API      -> http://127.0.0.1:$port  (queue backend: $backend)"
Start-Process -FilePath $python -WorkingDirectory (Join-Path $root 'backend') -ArgumentList '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', $port

if ($backend -eq 'redis') {
    Write-Host 'Worker   -> RQ worker (requires Redis at REDIS_URL)'
    Start-Process -FilePath $python -WorkingDirectory (Join-Path $root 'backend') -ArgumentList '-m', 'worker.main'
}

Write-Host 'Frontend -> http://127.0.0.1:5173'
Set-Location (Join-Path $root 'frontend')
if (-not (Test-Path 'node_modules')) { npm ci }
npm run dev
