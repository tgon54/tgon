@echo off
REM Export an AI-MoCap output folder to Unreal FBX (UE4 Mannequin + UE5 Manny)
REM  Drag the output folder onto this file, or: export_unreal.bat out_myvideo
cd /d "%~dp0"
set "OUT=%~1"
if "%OUT%"=="" set /p "OUT=Output folder: "
set "OUT=%OUT:"=%"
python -m aimocap.unreal "%OUT%" --target ue4
python -m aimocap.unreal "%OUT%" --target ue5
pause
