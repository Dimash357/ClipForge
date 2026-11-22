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
"%ENVPY%" tools\preflight.py all
"%ENVPY%" tools\preflight.py deps --pip-check
"%ENVPY%" manage.py diagnose
pause
