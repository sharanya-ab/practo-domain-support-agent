@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set MOCK_LLM=true
set CREWAI_DISABLE_TELEMETRY=true
set OTEL_SDK_DISABLED=true
uvicorn api.main:app --host 127.0.0.1 --port 8000
endlocal
