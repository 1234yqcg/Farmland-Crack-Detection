@echo off
chcp 65001 >nul
title 农田干裂程度识别系统 - Windows版

echo ========================================
echo   农田干裂程度识别系统 (Windows版)
echo ========================================
echo.

cd /d "%~dp0"

python gui\main_window.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序启动失败！
    echo.
    echo 请检查：
    echo   1. Python是否已安装 (需要Python 3.8+)
    echo   2. 是否已激活虚拟环境
    echo   3. 依赖包是否已安装: pip install -r requirements.txt
    echo.
    pause
)