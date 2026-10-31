@echo off
REM AI-MoCap launcher
REM  - Double-click: asks for the video (drag the file into the window, press Enter)
REM  - Or drag the video file onto run.bat
REM  - Or from cmd: run.bat video.mp4 1.75
cd /d "%~dp0"

set "VID=%~1"
set "H=%~2"
if not "%VID%"=="" goto havevid
echo.
echo ============================================================
echo   Drag your AI video file into this window, then press Enter
echo ============================================================
set /p "VID=Video: "
:havevid
set "VID=%VID:"=%"
if "%VID%"=="" (
  echo No video given.
  pause
  exit /b 1
)
if not exist "%VID%" (
  echo File not found: %VID%
  pause
  exit /b 1
)

if not "%H%"=="" goto haveh
set /p "H=Subject height in meters [Enter = 1.75]: "
:haveh
if "%H%"=="" set "H=1.75"

for %%F in ("%VID%") do set "OUT=%~dp0out_%%~nF"
echo.
echo Video : %VID%
echo Output: %OUT%
echo.

python -m aimocap "%VID%" "%OUT%" --height %H%
if errorlevel 1 (
  echo.
  echo *** ERROR - copy the messages above and send them ***
  pause
  exit /b 1
)

if exist "%OUT%\viewer.html" start "" "%OUT%\viewer.html"
if exist "%OUT%\report.html" start "" "%OUT%\report.html"
echo.
echo Done. Results in: %OUT%
pause
