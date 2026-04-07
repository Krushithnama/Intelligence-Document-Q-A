$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}

.\.venv\Scripts\activate
pip install -r .\requirements.txt

if (-not (Test-Path ".\backend\.env")) {
  Copy-Item ".\backend\.env.example" ".\backend\.env"
  Write-Host "Created backend\.env from example. Please set GEMINI_API_KEY."
}

uvicorn backend.api.main:app --reload --port 8000

