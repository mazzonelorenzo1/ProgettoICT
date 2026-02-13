@echo off
setlocal EnableExtensions
title Smart Peephole Demo (Smart Launcher)

REM =========================
REM CONFIG
REM =========================
set PORT=5000
set API_KEY=supersecret123
set ENV_NAME=peephole
REM =========================

cd /d "%~dp0"

echo ============================================
echo   Smart Peephole Demo - Smart Launcher
echo ============================================
echo Project: %cd%
echo.

REM ---------- choose python launcher ----------
set "PY=py"
where py >nul 2>nul
if %errorlevel% neq 0 set "PY=python"

echo Using: %PY%
echo.

REM ---------- (optional) conda detection ----------
set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"

REM ---------- find ngrok.exe (optional) ----------
set "NGROK="
if exist "%~dp0tools\ngrok.exe" set "NGROK=%~dp0tools\ngrok.exe"
if exist "%~dp0ngrok.exe" set "NGROK=%~dp0ngrok.exe"

REM ---------- start Flask ----------
echo [1/2] Starting Flask...

REM prefer local venv if present
set "VENV_ACT=%~dp0.venv\Scripts\activate.bat"

if exist "%VENV_ACT%" (
  echo Using venv: %VENV_ACT%
  start "Flask Server" cmd /k ^
    "call "%VENV_ACT%" ^&^& set PEEPHOLE_API_KEY=%API_KEY% ^&^& python -m face.app"
) else if defined CONDA_BAT (
  echo Using conda env: %ENV_NAME%
  start "Flask Server" cmd /k ^
    ""%CONDA_BAT%" activate %ENV_NAME% ^&^& set PEEPHOLE_API_KEY=%API_KEY% ^&^& python -m face.app"
) else (
  echo No venv/conda detected. Using system Python.
  start "Flask Server" cmd /k ^
    "set PEEPHOLE_API_KEY=%API_KEY% ^&^& %PY% -m face.app"
)


REM wait a bit
timeout /t 4 >nul

REM ---------- start ngrok if present ----------
echo [2/2] Public link:
if defined NGROK (
  echo Starting ngrok...
  start "ngrok Tunnel" cmd /k ^
    ""%NGROK%" http %PORT%"
  echo.
  echo Copy the https://... link from the ngrok window.
) else (
  echo ngrok.exe NOT found - running LOCAL ONLY.
  echo Local URL: http://127.0.0.1:%PORT%
  echo To enable a public link:
  echo   1) Download ngrok
  echo   2) Put ngrok.exe in this folder OR in tools\
)

echo.
echo API KEY for enroll: %API_KEY%
echo.
pause
