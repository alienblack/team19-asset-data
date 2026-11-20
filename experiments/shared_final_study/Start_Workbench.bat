@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python 3.12 environment first. Follow START_HERE.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_study.py --mode guided
