@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  where python >nul 2>&1
  if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
  )
  set "PYTHON_BOOTSTRAP=python"
) else (
  set "PYTHON_BOOTSTRAP=py -3.12"
)

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js not found. Install Node 18+ from https://nodejs.org/
  pause
  exit /b 1
)

if not exist "models" mkdir models

echo Preparing the runtime for this computer...
%PYTHON_BOOTSTRAP% scripts\bootstrap.py
if errorlevel 1 (
  echo Runtime setup failed. Read the error above and see the How to Use tab for fixes.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd /d "%~dp0frontend"
  npm install --strict-ssl=false --registry=https://registry.npmjs.org/ --no-audit --no-fund
  cd /d "%~dp0"
)

start "LLM Mind Visualizer API" cmd /k "cd /d "%~dp0backend" && .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "LLM Mind Visualizer UI" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 4 /nobreak >nul
start http://127.0.0.1:5173

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.
echo If the models/ folder is empty, download a HuggingFace model into it.
echo This window can be closed. Keep the two server windows open while using the app.
pause
