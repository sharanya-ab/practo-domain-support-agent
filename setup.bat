@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
set CREWAI_DISABLE_TELEMETRY=true
set OTEL_SDK_DISABLED=true
python scripts\download_embedding_model.py
if errorlevel 1 (
  echo Setup failed. Review the error above.
  exit /b 1
)
echo Setup complete. Run run_all.bat next.
endlocal
