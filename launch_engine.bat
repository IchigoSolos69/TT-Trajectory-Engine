@echo off
setlocal enabledelayedexpansion
title TT Trajectory Engine - Cold Boot Sequence
color 0B

set ROOT_DIR=C:\Users\Adi\Documents\GitHub\TT Trajectory Engine

echo =====================================================
echo    TABLE TENNIS TRAJECTORY ENGINE : COLD BOOT
echo =====================================================
echo.

echo [1/4] Terminating stale processes...
taskkill /F /IM tt_trajectory_engine.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1

echo [2/4] Select Execution Environment:
echo   [1] Simulation Mode (Dataset Video)
echo   [2] Live Mode (External Webcam + Homography)
set /p RUN_MODE="Enter 1 or 2: "

:: Force High-Performance NVIDIA GPU Execution
set CUDA_VISIBLE_DEVICES=0
set CUDA_MODULE_LOADING=EAGER

cd /d "%ROOT_DIR%\backend_cpp" || (
    color 0C
    echo [FATAL ERROR] backend_cpp directory missing.
    pause
    exit /b 1
)

if not exist ".\build\Release\tt_trajectory_engine.exe" (
    color 0C
    echo [FATAL ERROR] C++ executable not found. Compile it using CMake first.
    pause
    exit /b 1
)

if "%RUN_MODE%"=="2" (
    echo [INFO] Booting Live Production Environment...
    start "TT Engine - C++ Backend (Port 9002)" cmd /k ".\build\Release\tt_trajectory_engine.exe --live"
) else (
    echo [INFO] Booting Simulation Environment...
    start "TT Engine - C++ Backend (Port 9002)" cmd /k ".\build\Release\tt_trajectory_engine.exe --sim"
)

echo [3/4] Initializing Next.js Frontend...
:: Give the GPU a 3-second head start to deserialize the TensorRT model
timeout /t 3 /nobreak >nul

set FRONTEND_DIR=
if exist "%ROOT_DIR%\package.json" (
    set FRONTEND_DIR=%ROOT_DIR%
) else if exist "%ROOT_DIR%\frontend\package.json" (
    set FRONTEND_DIR=%ROOT_DIR%\frontend
) else if exist "%ROOT_DIR%\dashboard\package.json" (
    set FRONTEND_DIR=%ROOT_DIR%\dashboard
)

if "%FRONTEND_DIR%"=="" (
    color 0C
    echo [FATAL ERROR] Could not locate Next.js package.json.
    pause
    exit /b 1
)

cd /d "%FRONTEND_DIR%" || (
    color 0C
    echo [FATAL ERROR] Could not access frontend directory.
    pause
    exit /b 1
)

start "TT Engine - React Frontend (Port 3000)" cmd /k "npm run dev"

echo [4/4] Launching Dashboard...
:: Give the React server 5 seconds to compile before opening the browser
timeout /t 5 /nobreak >nul
start http://localhost:3000
exit