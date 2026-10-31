@echo off
REM AI-MoCap - install / repair all required packages
cd /d "%~dp0"
python -m aimocap.setup_env
echo.
pause
