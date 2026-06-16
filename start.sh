#!/usr/bin/env bash
# Media Prompter — Startup Script
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   Media Prompter                          ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

if ! command -v python3 &>/dev/null; then
  echo "ERROR: Python 3 not found. Please install Python 3.9+"
  exit 1
fi

echo "Python: $(python3 --version)"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

export PYTHONNOUSERSITE=1

echo "Installing dependencies (this may take a while first time)..."
.venv/bin/python3 -m pip install -q -r requirements.txt

echo "Dependencies installed"
echo ""
PORT="${PORT:-6666}"
echo "Starting Media Prompter at http://localhost:${PORT}"
echo "Opening browser..."
echo ""

(
  while ! curl -s "http://localhost:${PORT}/health" > /dev/null; do
    sleep 1
  done
  open "http://localhost:${PORT}"
) &

export PORT="${PORT:-6666}"
cd backend
../.venv/bin/python3 main.py
