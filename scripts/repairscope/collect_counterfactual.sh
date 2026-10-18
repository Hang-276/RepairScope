#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
rs_cf_plan_args=()
if [[ -n "${PROBE_PLANS:-}" ]]; then rs_cf_plan_args=(--plans "$PROBE_PLANS"); fi
"$RS_PYTHON" -m repairscope counterfactual --config "$RS_CONFIG" \
  ${rs_cf_plan_args[@]+"${rs_cf_plan_args[@]}"} --set environment.offline_audit=true \
  --output-dir "${CF_OUTPUT:-$RS_RUN_ROOT/counterfactual}" ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
