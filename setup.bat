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

echo === ClipForge Studio setup (environment: %ENVDIR%) ===
if exist "%ENVPY%" goto have_env
if exist "%ENVDIR%\" goto broken_env
call :find_python
if not defined BASEPY goto no_python
echo Creating virtual environment %ENVDIR% with: %BASEPY%
%BASEPY% -m venv "%ENVDIR%"
if not exist "%ENVPY%" goto venv_failed
:have_env
echo [OK] Using %ENVPY%

rem ---- only the env's own python is used below: nothing is ever installed into the system Python ----
"%ENVPY%" tools\preflight.py venv || goto failed
"%ENVPY%" tools\preflight.py ffmpeg || goto failed
"%ENVPY%" tools\preflight.py constraints "%ENVDIR%\clipforge-constraints.txt" || goto failed
echo Installing requirements into the environment...
"%ENVPY%" -m pip install --disable-pip-version-check -r requirements.txt -c "%ENVDIR%\clipforge-constraints.txt"
if errorlevel 1 goto pip_failed
"%ENVPY%" tools\preflight.py deps --pip-check || goto failed
"%ENVPY%" tools\preflight.py backup-db || goto failed
"%ENVPY%" manage.py migrate || goto failed
if not exist source_library mkdir source_library
echo.
echo [OK] Done. Start the app with start.bat
pause
exit /b 0

:pip_failed
echo [X]  pip could not install the requirements (details above).
echo      If the kept NumPy version is the reason, allow pip to change it:
echo        set CLIPFORGE_ALLOW_NUMPY_UPGRADE=1
echo      and run setup.bat again from the same window.
goto failed

:broken_env
echo [X]  Folder %ENVDIR% exists but has no Scripts\python.exe. Nothing was deleted.
echo      Rename or remove that folder yourself, then run setup.bat again.
goto failed

:no_python
echo [X]  Python 3.10+ not found (py launcher or python in PATH).
echo      Install Python from python.org and tick "Add python.exe to PATH".
goto failed

:venv_failed
echo [X]  Could not create the virtual environment. The system Python was not changed.
goto failed

:find_python
set "BASEPY="
py -3.10 -c "import sys" >nul 2>nul && set "BASEPY=py -3.10" && exit /b 0
py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set "BASEPY=py -3" && exit /b 0
python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set "BASEPY=python" && exit /b 0
exit /b 0

:failed
echo.
echo [X]  Stopped - see the message above. Your projects, database and videos were not changed.
pause
exit /b 1
