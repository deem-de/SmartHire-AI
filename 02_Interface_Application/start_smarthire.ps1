$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m uvicorn app_server:app --host 127.0.0.1 --port 3001
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m uvicorn app_server:app --host 127.0.0.1 --port 3001
} else {
    Write-Error "Python was not found on this machine."
}
