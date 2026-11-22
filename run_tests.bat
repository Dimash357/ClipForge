@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [!] .venv not found. Run setup.bat first. & pause & exit /b 1)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install -r requirements-dev.txt
rem set CLIPFORGE_SKIP_SLOW=1 to skip the ffmpeg end-to-end tests (~4 min)
"%VPY%" manage.py test studio -v 1
pause
