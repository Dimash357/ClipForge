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
"%ENVPY%" tools\preflight.py venv || goto failed
"%ENVPY%" tools\preflight.py ffmpeg || goto failed
"%ENVPY%" tools\preflight.py constraints "%ENVDIR%\clipforge-constraints.txt" || goto failed
rem test-only dependency (scikit-image: public-domain test face); the env's NumPy is kept
"%ENVPY%" -m pip install --disable-pip-version-check -r requirements-dev.txt -c "%ENVDIR%\clipforge-constraints.txt" || goto failed
rem set CLIPFORGE_SKIP_SLOW=1 to skip the ffmpeg end-to-end tests (~5 min). Tests use a temporary database.
"%ENVPY%" manage.py test studio -v 1
pause
exit /b 0

:failed
echo.
echo [X]  Stopped - see the message above. Your projects, database and videos were not changed.
pause
exit /b 1
