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
if exist "%~dp0ai_env.bat" call "%~dp0ai_env.bat"
echo === ClipForge AI: Ollama (LLM), ComfyUI (images), Piper (voice) ===
echo Before every download or install you are asked first. Safe to run again: nothing is downloaded twice.
echo   install_ai.bat           install / update step by step
echo   install_ai.bat check     only check the providers
echo   install_ai.bat --comfy-dir "D:\ComfyUI_windows_portable"   use an existing ComfyUI
echo.
"%ENVPY%" tools\preflight.py venv || goto failed
"%ENVPY%" tools\preflight.py constraints "%ENVDIR%\clipforge-constraints.txt" || goto failed
"%ENVPY%" tools\install_ai.py %*
if errorlevel 1 goto problems
echo.
echo [OK] Done. Start ClipForge with start.bat - it also starts Ollama / ComfyUI when needed.
pause
exit /b 0

:problems
echo.
echo [!]  Some providers are not ready - see the messages above. ClipForge still works without them;
echo      fix the item and run install_ai.bat again (finished steps and downloads are kept).
pause
exit /b 1

:failed
echo.
echo [X]  Stopped - see the message above. Your projects, database and videos were not changed.
pause
exit /b 1
