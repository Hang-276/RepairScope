#!/usr/bin/env bash
set -euo pipefail
RS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for rs_name in a0_actor_only a1_dynamics a2_paired_effect a3_full a4_no_boundary a5_no_escape; do
  bash "$RS_SCRIPT_DIR/$rs_name.sh" "$@"
done
