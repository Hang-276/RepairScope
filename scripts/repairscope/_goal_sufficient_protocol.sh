#!/usr/bin/env bash
# Shared arguments for the current v2 main and exploration experiments.
set -euo pipefail
if [[ "${SEEDS:-100}" != 100 ]]; then
  echo 'The initial v2 protocol is fixed to SEEDS=100.' >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo 'Set RUN_ROOT to a new directory whose name contains goal_sufficient or v2.' >&2
  exit 2
fi
RS_PROTOCOL_NAME="${RUN_ROOT%/}"
RS_PROTOCOL_NAME="${RS_PROTOCOL_NAME##*/}"
if [[ "$RS_PROTOCOL_NAME" != *goal_sufficient* && "$RS_PROTOCOL_NAME" != *v2* ]]; then
  echo 'RUN_ROOT must explicitly identify this new v2 or goal_sufficient run.' >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo 'RUN_ROOT already exists. Choose a fresh directory to preserve prior runs.' >&2
  exit 2
fi
if [[ -z "${TRAIN_DATA:-}" || -z "${EVAL_DATA:-}" ]]; then
  echo 'Set TRAIN_DATA and EVAL_DATA to the shared official train and held-out dev data.' >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != 1 && ( ! -f "$TRAIN_DATA" || ! -f "$EVAL_DATA" ) ]]; then
  echo 'The shared TRAIN_DATA and EVAL_DATA files must already exist.' >&2
  exit 2
fi
SEEDS=100
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
# The official initial comparison has one fixed training schedule and scope.
RS_PROTOCOL_ARGS=(--set train.epochs=30 --set train.max_steps=null
  --set train.horizon=4 --set model.scope_budget=128 --set model.candidate_limit=256
  --set "data.train=$TRAIN_DATA" --set "data.dev=$EVAL_DATA"
  --set probing.budget_calls=2000)

rs_check_protocol_arguments() {
  local rs_argument
  for rs_argument in "$@"; do
    case "$rs_argument" in
      --variant|--variant=*|--config|--config=*|--seed|--seed=*|--output-dir|--output-dir=*|--data|--data=*|--checkpoint|--checkpoint=*|--plans|--plans=*)
        echo "Protocol-owned argument cannot be overridden: $rs_argument" >&2; exit 2 ;;
      model.*|loss.*|train.*|variant=*|data.train=*|data.dev=*|probing.budget_calls=*|collection.*|--set=model.*|--set=loss.*|--set=train.*|--set=variant=*|--set=data.train=*|--set=data.dev=*|--set=probing.budget_calls=*|--set=collection.*)
        echo "Protocol-owned setting cannot be overridden: $rs_argument" >&2; exit 2 ;;
    esac
  done
}
