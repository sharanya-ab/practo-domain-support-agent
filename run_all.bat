@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set MOCK_LLM=true
set CREWAI_DISABLE_TELEMETRY=true
set OTEL_SDK_DISABLED=true
python run_all.py
if errorlevel 1 echo One or more stages failed. See output above and artifacts.
endlocal
