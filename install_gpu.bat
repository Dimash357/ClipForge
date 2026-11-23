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
echo === NVIDIA CUDA 12 libraries for Whisper on the GPU (about 1.3 GB, into %ENVDIR% only) ===
"%ENVPY%" tools\preflight.py venv || goto failed
"%ENVPY%" tools\preflight.py constraints "%ENVDIR%\clipforge-constraints.txt" || goto failed
rem cuBLAS 12.9.2.10 is the newest cuBLAS 12 wheel built for Windows; cuDNN 9 is what CTranslate2 4.x expects
"%ENVPY%" -m pip install --disable-pip-version-check "nvidia-cublas-cu12==12.9.2.10" "nvidia-cudnn-cu12>=9.1,<10" -c "%ENVDIR%\clipforge-constraints.txt" || goto failed
"%ENVPY%" tools\preflight.py gpu
echo.
echo [OK] Done. Restart start.bat: Whisper will use the GPU (see the System page).
pause
exit /b 0

:failed
echo.
echo [X]  Stopped - see the message above. ClipForge keeps working on the CPU.
pause
exit /b 1
