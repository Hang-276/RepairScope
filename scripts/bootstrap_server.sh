#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11/3.12"'
command -v git-lfs >/dev/null || { echo "Install git-lfs first; no system configuration was changed." >&2; exit 1; }
git submodule update --init --recursive
git -C vendor/appworld lfs install --local
git -C vendor/appworld lfs pull
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip<26'
.venv/bin/python -m pip install -e '.[ml,cuda,test]'
.venv/bin/python -m pip install -e vendor/appworld -e 'vendor/appworld/experiments[simplified]'
(
  cd vendor/appworld
  ../../.venv/bin/appworld install --repo
  # The upstream downloader deletes an existing data directory. Reuse only
  # after our manifest check; never let bootstrap erase previous experiments.
  if [[ ! -d data ]]; then
    APPWORLD_ROOT="$PWD" ../../.venv/bin/appworld download data --version 0.2.0 --mode minimal
  fi
)
mkdir -p runs/environment
.venv/bin/repairscope manifest --config configs/appworld.toml --output runs/environment/dataset_manifest.json
"$PYTHON_BIN" -m venv .venv-actor
.venv-actor/bin/python -m pip install --upgrade 'pip<26'
.venv-actor/bin/python -m pip install 'vllm==0.11.0'
.venv/bin/python -m pip freeze > runs/environment/main_requirements.txt
.venv-actor/bin/python -m pip freeze > runs/environment/actor_requirements.txt
.venv/bin/repairscope doctor --config configs/appworld.toml --output runs/environment/doctor.json
.venv/bin/python -m pytest -q
echo 'Next: .venv/bin/python scripts/prepare_models.py --download'
