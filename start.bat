@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
rem FFmpeg may also live inside the project: ffmpeg\bin\ffmpeg.exe (then no PATH change is needed)
if exist "%~dp0ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0ffmpeg\bin;%PATH%"
rem Virtual environment = the project's "env" folder (override: set CLIPFORGE_ENV=C:\path\to\env)
set "ENVDIR=%~dp0env"
if defined CLIPFORGE_ENV set "ENVDIR=%CLIPFORGE_ENV%"
set "ENVPY=%ENVDIR%\Scripts\python.exe"
if exist "%ENVPY%" goto env_ok
echo [X]  Virtual environment not found: %ENVDIR%
echo      Run setup.bat first.
pause
exit /b 1
:env_ok
"%ENVPY%" tools\preflight.py all || goto failed
"%ENVPY%" tools\preflight.py backup-db || goto failed
"%ENVPY%" manage.py migrate || goto failed
rem --noreload: a code change or editor save must never restart the server in the middle of a render.
echo Opening http://127.0.0.1:8000   (stop the server with Ctrl+C)
start "" http://127.0.0.1:8000
"%ENVPY%" manage.py runserver 127.0.0.1:8000 --noreload
pause
exit /b 0

:failed
echo.
echo [X]  Stopped - see the message above. Your projects, database and videos were not changed.
pause
exit /b 1
