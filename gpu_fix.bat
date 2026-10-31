@echo off
REM AI-MoCap - check and repair NVIDIA GPU support
cd /d "%~dp0"
python -m aimocap.gpu_check --fix
echo.
pause
