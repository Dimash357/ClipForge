@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [!] .venv not found. Run setup.bat first. & pause & exit /b 1)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" manage.py migrate
rem --noreload: a code change or editor save must never restart the server in the middle of a render.
start "" http://127.0.0.1:8000
"%VPY%" manage.py runserver 127.0.0.1:8000 --noreload
pause
