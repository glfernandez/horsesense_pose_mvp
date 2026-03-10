#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PORT="${PORT:-8504}"
python -m streamlit run app/dashboard.py --server.port "$PORT" --server.headless true
