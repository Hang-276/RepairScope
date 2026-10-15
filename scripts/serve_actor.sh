#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f runs/models.lock.json || { echo 'Run scripts/prepare_models.py first.' >&2; exit 1; }
ACTOR_PATH="$(.venv/bin/python -c 'import json; d=json.load(open("runs/models.lock.json"))["actor"]; print(d["local_path"] or d["repo_id"])')"
ACTOR_REV="$(.venv/bin/python -c 'import json; print(json.load(open("runs/models.lock.json"))["actor"]["revision"])')"
# Foreground process: caller records its own PID if backgrounding. No pkill or
# global environment changes. Use another free port via ACTOR_PORT + config.
exec .venv-actor/bin/vllm serve "$ACTOR_PATH" \
  --revision "$ACTOR_REV" --served-model-name Qwen/Qwen3-8B-AWQ \
  --host 127.0.0.1 --port "${ACTOR_PORT:-8000}" \
  --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization "${ACTOR_GPU_UTIL:-0.35}" \
  --dtype half --enforce-eager --generation-config vllm
