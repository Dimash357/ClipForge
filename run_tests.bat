@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate
pip install -r requirements-dev.txt
rem set CLIPFORGE_SKIP_SLOW=1 to skip the ffmpeg end-to-end tests (~3 min)
python manage.py test studio -v 1
pause
