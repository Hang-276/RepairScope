#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
"$RS_PYTHON" -m repairscope smoke --config configs/repairscope/local_smoke.yaml \
  --output-dir "${SMOKE_OUTPUT:-$RS_RUN_ROOT/local_smoke}" ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
