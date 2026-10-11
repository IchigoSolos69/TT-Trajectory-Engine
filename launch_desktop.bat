@echo off
setlocal
set "DESKTOP_EXE=%~dp0desktop\dist\TT Trajectory Engine paired training v2\TT Trajectory Engine.exe"
if not exist "%DESKTOP_EXE%" set "DESKTOP_EXE=%~dp0desktop\dist\TT Trajectory Engine paired training v1\TT Trajectory Engine.exe"
if not exist "%DESKTOP_EXE%" set "DESKTOP_EXE=%~dp0desktop\dist\TT Trajectory Engine training review v2\TT Trajectory Engine.exe"
if not exist "%DESKTOP_EXE%" set "DESKTOP_EXE=%~dp0desktop\dist\TT Trajectory Engine\TT Trajectory Engine.exe"
if exist "%DESKTOP_EXE%" (
    start "" "%DESKTOP_EXE%"
    exit /b 0
)
if not exist "%~dp0desktop\node_modules\electron\dist\electron.exe" (
    echo Build the desktop application first with tools\build_desktop.ps1.
    pause
    exit /b 1
)
start "" "%~dp0desktop\node_modules\electron\dist\electron.exe" "%~dp0desktop"
