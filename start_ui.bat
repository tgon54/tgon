@echo off
REM AI-MoCap - local web interface
cd /d "%~dp0"
python -c "import rtmlib, onnxruntime, cv2, scipy, matplotlib" 2>nul
if errorlevel 1 (
  echo First run: installing required packages...
  python -m aimocap.setup_env
  if errorlevel 1 ( echo *** Installation failed - send a screenshot *** & pause & exit /b 1 )
)
echo Starting AI-MoCap interface...
python -m aimocap.server
pause
