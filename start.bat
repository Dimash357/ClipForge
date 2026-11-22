@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate
python manage.py migrate
rem --noreload: a code change or editor save must never restart the server in the middle of a render.
start "" http://127.0.0.1:8000
python manage.py runserver 127.0.0.1:8000 --noreload
pause
