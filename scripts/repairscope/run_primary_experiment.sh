#!/usr/bin/env bash
set -euo pipefail
# Current authorized scope: one paired primary experiment, A1 versus A3, seed 100.
if [[ "${SEEDS:-100}" != 100 ]]; then
  echo 'This stage uses only SEEDS=100. Other seeds need a later experiment decision.' >&2
  exit 2
fi
if [[ -n "${TASK_FILE:-}" ]]; then
  echo 'The primary experiment uses the complete official test_normal split; unset TASK_FILE.' >&2
  exit 2
fi
SEEDS=100
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
rs_experiment dynamics a1_dynamics "$@" --set evaluation.split=test_normal --set 'evaluation.task_ids=[]'
rs_experiment full a3_full "$@" --set evaluation.split=test_normal --set 'evaluation.task_ids=[]'
