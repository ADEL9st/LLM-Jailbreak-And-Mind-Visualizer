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
%PYTHON_BOOTSTRAP% -u scripts\bootstrap.py
if errorlevel 1 (
  echo Runtime setup failed. Read the error above and see the How to Use tab for fixes.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd /d "%~dp0frontend"
  npm.cmd install --registry=https://registry.npmjs.org/ --no-audit --no-fund
  if errorlevel 1 (
    echo Frontend setup failed. Check Node.js, npm, and your network connection.
    pause
    exit /b 1
  )
  cd /d "%~dp0"
)

call :port_listening 8000
if errorlevel 1 (
  echo Starting backend on http://127.0.0.1:8000 ...
  start "LLM Mind Visualizer API" /D "%~dp0backend" cmd.exe /k ".\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
) else (
  echo Backend is already running on port 8000.
)

call :port_listening 5173
if errorlevel 1 (
  echo Starting frontend on http://127.0.0.1:5173 ...
  start "LLM Mind Visualizer UI" /D "%~dp0frontend" cmd.exe /k "npm.cmd run dev"
) else (
  echo Frontend is already running on port 5173.
)

if /I "%LLM_MIND_NO_BROWSER%"=="1" (
  echo Browser launch skipped.
) else (
  echo Waiting for the interface...
  powershell.exe -NoProfile -Command "$deadline = (Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { try { $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5173' -TimeoutSec 2; if ($response.StatusCode -lt 500) { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"
  if errorlevel 1 (
    echo The interface did not answer within 30 seconds. Check the API and UI windows for an error.
  ) else (
    start "" "http://127.0.0.1:5173"
  )
)

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.
echo If the models/ folder is empty, download a HuggingFace model into it.
echo This window can be closed. Keep the two server windows open while using the app.
if /I "%LLM_MIND_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:port_listening
netstat -ano | findstr /R /C:":%~1 .*LISTENING" >nul
exit /b %errorlevel%
