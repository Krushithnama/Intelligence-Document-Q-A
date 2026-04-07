$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

.\.venv\Scripts\activate
streamlit run .\frontend\app.py

