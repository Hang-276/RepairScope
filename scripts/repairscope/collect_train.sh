#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
"$RS_PYTHON" -m repairscope collect --config "$RS_CONFIG" --split train \
  --set environment.offline_audit=true --output-dir "${COLLECT_OUTPUT:-$RS_RUN_ROOT/natural}" \
  ${RS_TASK_ARGS[@]+"${RS_TASK_ARGS[@]}"} ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
