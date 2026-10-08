@echo off
setlocal
title TT Trajectory Engine
color 0B
echo =====================================================
echo          TABLE TENNIS TRAJECTORY ENGINE
echo =====================================================
echo   [1] Simulation - analyze a dataset or local recording
echo   [2] Live - camera starts only from the dashboard
choice /c 12 /n /m "Enter 1 or 2: "
if errorlevel 3 exit /b 1
if errorlevel 2 (set "ENGINE_MODE=live") else (set "ENGINE_MODE=simulation")
set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_MODULE_LOADING=EAGER"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch_services.ps1" -Mode %ENGINE_MODE%
if errorlevel 1 (
    echo Launch failed. See the message above and artifacts\launcher_* logs.
    pause
    exit /b 1
)
exit /b 0
