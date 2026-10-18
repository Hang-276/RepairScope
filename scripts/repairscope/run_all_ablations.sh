#!/usr/bin/env bash
set -euo pipefail
RS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for rs_name in a4_no_boundary a5_no_escape ablate_static_scope ablate_no_learned_edges ablate_no_temporal ablate_unified ablate_no_rollout ablate_no_goal; do
  bash "$RS_SCRIPT_DIR/$rs_name.sh" "$@"
done
