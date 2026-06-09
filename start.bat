@echo off
chcp 65001 >nul
title AI一照成证

echo.
echo  ╔══════════════════════════════════════╗
echo  ║     AI一照成证 — 一键启动            ║
echo  ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/2] 启动 API 服务器 (端口 8000)...
start "AI一照成证-API" python -X utf8 api_server.py --port 8000

timeout /t 4 /nobreak >nul

echo [2/2] 打开浏览器...
start http://127.0.0.1:8000/app

echo.
echo  前端界面: http://127.0.0.1:8000/app
echo  API 文档:  http://127.0.0.1:8000/docs
echo  停止服务:  关闭本窗口或 Ctrl+C
echo.
pause
