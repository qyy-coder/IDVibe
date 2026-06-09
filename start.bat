@echo off
chcp 65001 >nul
title AI一照成证
cd /d "%~dp0"

echo.
echo   AI一照成证 - 一键启动
echo   =====================
echo.

:: 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo        https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 检查依赖
python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt -q
)

echo [1/2] 启动 API 服务器...
start "AI一照成证-API" /MIN python -X utf8 api_server.py --port 8000
echo        等待服务就绪...

:: 等待服务启动
:waitloop
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1
if %errorlevel% equ 0 goto ready
echo [错误] 服务器启动超时，请手动运行: python api_server.py --port 8000
pause
exit /b 1

:ready
echo [2/2] 打开浏览器...
start http://127.0.0.1:8000/app

echo.
echo   =====================
echo   前端: http://127.0.0.1:8000/app
echo   文档: http://127.0.0.1:8000/docs
echo   停止: 关闭 API 窗口或 Ctrl+C
echo   =====================
echo.

:: 3秒后自动关闭本窗口
timeout /t 3 /nobreak >nul
exit
