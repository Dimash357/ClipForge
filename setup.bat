@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (echo [!] ffmpeg not found in PATH. Install FFmpeg first, see README.md & pause & exit /b 1)

rem --- find a Python 3.10+ to create the virtual environment (python.org 3.12 is recommended) ---
set "PY="
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul && set "PY=python"
if not defined PY (echo [!] Python 3.10+ not found. Install Python 3.12 from python.org with "Add to PATH". & pause & exit /b 1)

rem --- a half-created .venv (no python.exe inside) is rebuilt ---
if exist .venv if not exist .venv\Scripts\python.exe rmdir /s /q .venv
if not exist .venv\Scripts\python.exe (
    echo Creating .venv with: %PY%
    %PY% -m venv .venv
)
if not exist .venv\Scripts\python.exe (echo [!] Could not create .venv - nothing was installed. & pause & exit /b 1)

rem --- everything below uses the venv's own python: packages never go into the system Python ---
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -c "import sys; print('Using', sys.executable, sys.version.split()[0])"
"%VPY%" -m pip install --upgrade pip || (pause & exit /b 1)
"%VPY%" -m pip install -r requirements.txt || (echo [!] pip install failed & pause & exit /b 1)
"%VPY%" manage.py migrate || (echo [!] migrate failed & pause & exit /b 1)
if not exist source_library mkdir source_library
echo.
echo Done. Start the app with start.bat
pause
