@echo off
chcp 65001 >nul
title 极限训练脚本 - Windows版

echo ========================================
echo   YOLOv10 极限训练脚本 (Windows版)
echo ========================================
echo.

cd /d "%~dp0"

python scripts\train_extreme.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 训练失败！
    echo.
    pause
)