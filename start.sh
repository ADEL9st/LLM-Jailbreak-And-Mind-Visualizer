#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Install Python 3.10+ first."
  exit 1
fi

if ! python3 -c "import venv" >/dev/null 2>&1; then
  echo "Python venv module missing."
  echo "On Debian/Ubuntu run: sudo apt install python3-venv"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install Node 18+ first."
  exit 1
fi

echo "Preparing the runtime for this computer..."
python3 scripts/bootstrap.py

source backend/.venv/bin/activate

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  cd frontend && npm install --no-audit --no-fund -q && cd "$ROOT"
fi

mkdir -p models

echo ""
echo "Starting backend on http://127.0.0.1:8000 ..."
cd "$ROOT/backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://127.0.0.1:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

sleep 3

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://127.0.0.1:5173 >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open http://127.0.0.1:5173 >/dev/null 2>&1 &
else
  echo "Open http://127.0.0.1:5173 in your browser."
fi

echo ""
echo "Backend:  http://127.0.0.1:8000  (PID $BACKEND_PID)"
echo "Frontend: http://127.0.0.1:5173  (PID $FRONTEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
