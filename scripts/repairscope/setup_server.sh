#!/usr/bin/env bash
set -euo pipefail
RS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RS_ROOT"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer targets the Linux experiment server; use local_smoke.sh on macOS.' >&2
  exit 2
fi
RS_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3.11}"
"$RS_BOOTSTRAP_PYTHON" -m venv .venv
RS_PYTHON="$RS_ROOT/.venv/bin/python"
"$RS_PYTHON" -m pip install --upgrade pip
"$RS_PYTHON" -m pip install -e '.[test,appworld]'
"$RS_PYTHON" -m pip freeze > "${SERVER_LOCKFILE:-requirements-server-resolved.txt}"
echo 'Dependencies installed. Configure APPWORLD_ROOT, public observation/goal maps and actor endpoint before collection.'
echo 'AppWorld data installation and any GPU experiment are separate server operations.'
