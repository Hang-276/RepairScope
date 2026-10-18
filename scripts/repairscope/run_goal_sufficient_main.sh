#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_goal_sufficient_protocol.sh"
rs_check_protocol_arguments "$@"
# Shared data are collected once before this script. No collection or probing here.
# Labels a4_no_boundary/a5_no_escape remain reserved for historical v1.1 runs.
rs_experiment actor_only a0_actor_only "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment dynamics a1_dynamics "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment paired_effect a2_paired_effect "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment full a3_full "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment goal_sufficient a4_goal_sufficient "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment no_goal_sufficiency v2_ablate_no_goal_sufficiency "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment v2_no_boundary v2_ablate_no_boundary "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment no_goal_relevant_escape v2_ablate_no_goal_relevant_escape "$@" "${RS_PROTOCOL_ARGS[@]}"
rs_experiment v2_static_scope v2_ablate_static_scope "$@" "${RS_PROTOCOL_ARGS[@]}"
