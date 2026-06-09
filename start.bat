@echo off
title AI ID Photo
cd /d "%~dp0"

echo.
echo   AI ID Photo - Starting...
echo   ==========================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt -q
)

echo [1/2] Starting API server on port 8000...
start "AI-IDPhoto-Server" /MIN python -X utf8 api_server.py --port 8000
echo        Waiting for server...

:waitloop
ping -n 3 127.0.0.1 >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
ping -n 3 127.0.0.1 >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
ping -n 3 127.0.0.1 >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
echo [ERROR] Server failed to start. Run manually: python api_server.py --port 8000
pause
exit /b 1

:ready
echo [2/2] Opening browser...
start http://127.0.0.1:8000/app

echo.
echo   ==========================
echo   Frontend: http://127.0.0.1:8000/app
echo   API Docs: http://127.0.0.1:8000/docs
echo   Stop:     Close the API server window
echo   ==========================
echo.

ping -n 3 127.0.0.1 >nul
exit
