#!/usr/bin/env bash
# Ours-only formal entry. This intentionally does not run the multi-variant
# main script or the B probing script.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_goal_sufficient_protocol.sh"
rs_check_protocol_arguments "$@"
rs_experiment goal_sufficient a4_goal_sufficient "$@" "${RS_PROTOCOL_ARGS[@]}"
