@echo off
chcp 65001 >nul
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (echo [!] ffmpeg not found in PATH. Install FFmpeg first, see README.md & pause & exit /b 1)
if not exist .venv (py -3.12 -m venv .venv || py -3 -m venv .venv)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1
python manage.py migrate || exit /b 1
if not exist source_library mkdir source_library
echo.
echo Done. Start the app with start.bat
pause
