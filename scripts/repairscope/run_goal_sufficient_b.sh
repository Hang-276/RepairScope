#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_goal_sufficient_protocol.sh"
rs_check_protocol_arguments "$@"
if [[ -z "${CHECKPOINT:-}" || -z "${PROBE_PLANS:-}" ]]; then
  echo 'Set CHECKPOINT to the completed A4 v2 checkpoint and PROBE_PLANS to shared train-only plans.' >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != 1 && ( ! -f "$CHECKPOINT" || ! -f "$PROBE_PLANS" ) ]]; then
  echo 'CHECKPOINT and PROBE_PLANS must already exist.' >&2
  exit 2
fi
# Deliberately separate from stage A; all strategies start from the same A4.
for rs_strategy in random uncertainty novelty_learnability risk; do
  RS_PROBE_LABEL="v2_probe_$rs_strategy" rs_probe "$rs_strategy" --variant goal_sufficient \
    "$@" "${RS_PROTOCOL_ARGS[@]}"
done
