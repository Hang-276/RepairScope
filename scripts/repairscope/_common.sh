#!/usr/bin/env bash
set -euo pipefail
RS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RS_ROOT"
if [[ -n "${REPAIRSCOPE_PYTHON:-}" ]]; then
  RS_PYTHON="$REPAIRSCOPE_PYTHON"
elif [[ -x "$RS_ROOT/.venv/bin/python" ]]; then
  RS_PYTHON="$RS_ROOT/.venv/bin/python"
else
  RS_PYTHON=python3
fi
RS_RUN_ROOT="${RUN_ROOT:-$RS_ROOT/runs/repairscope}"
RS_CONFIG="${CONFIG:-$RS_ROOT/configs/repairscope/base.yaml}"
read -r -a RS_SEEDS <<< "${SEEDS:-100 101 102}"
RS_DRY_ARGS=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then RS_DRY_ARGS+=(--dry-run); fi
RS_TASK_ARGS=()
if [[ -n "${TASK_FILE:-}" ]]; then RS_TASK_ARGS+=(--task-file "$TASK_FILE"); fi
RS_DATA_ARGS=()
if [[ -n "${TRAIN_DATA:-}" ]]; then RS_DATA_ARGS+=(--data "$TRAIN_DATA"); fi

# Train and evaluate every configured seed. All variants reuse the same input data.
rs_experiment() {
  local variant="$1" label="$2" rs_seed rs_dir
  local -a rs_calibration_args
  shift 2
  for rs_seed in "${RS_SEEDS[@]}"; do
    rs_dir="$RS_RUN_ROOT/$label/seed_$rs_seed"
    if [[ "$variant" == actor_only ]]; then
      "$RS_PYTHON" -m repairscope evaluate --config "$RS_CONFIG" --variant actor_only \
        --seed "$rs_seed" --output-dir "$rs_dir/evaluation" \
        ${RS_TASK_ARGS[@]+"${RS_TASK_ARGS[@]}"} ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
    else
      rs_calibration_args=()
      "$RS_PYTHON" -m repairscope train --config "$RS_CONFIG" --variant "$variant" \
        --seed "$rs_seed" --output-dir "$rs_dir/training" \
        ${RS_DATA_ARGS[@]+"${RS_DATA_ARGS[@]}"} ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
      if [[ -n "${EVAL_DATA:-}" ]]; then
        "$RS_PYTHON" -m repairscope predict-eval --config "$RS_CONFIG" --variant "$variant" \
          --checkpoint "$rs_dir/training/checkpoint.pt" --seed "$rs_seed" \
          --data "$EVAL_DATA" --output-dir "$rs_dir/prediction" \
          ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
        "$RS_PYTHON" -m repairscope calibrate --config "$RS_CONFIG" --variant "$variant" \
          --checkpoint "$rs_dir/training/checkpoint.pt" --seed "$rs_seed" \
          --predictions "$rs_dir/prediction/predictions.jsonl" \
          --output-dir "$rs_dir/calibration" ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
        rs_calibration_args=(--set "controller.calibration_file=$rs_dir/calibration/calibration.json")
      fi
      "$RS_PYTHON" -m repairscope evaluate --config "$RS_CONFIG" --variant "$variant" \
        --checkpoint "$rs_dir/training/checkpoint.pt" --seed "$rs_seed" \
        --output-dir "$rs_dir/evaluation" ${RS_TASK_ARGS[@]+"${RS_TASK_ARGS[@]}"} \
        ${rs_calibration_args[@]+"${rs_calibration_args[@]}"} ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
    fi
  done
}

rs_probe() {
  local strategy="$1" rs_seed rs_dir rs_checkpoint rs_calibration
  local rs_label="${RS_PROBE_LABEL:-probe_$1}"
  local -a rs_probe_calibration_args rs_plan_args rs_probe_data_args
  shift
  rs_plan_args=()
  if [[ -n "${PROBE_PLANS:-}" ]]; then rs_plan_args=(--plans "$PROBE_PLANS"); fi
  rs_probe_data_args=()
  if [[ -n "${TRAIN_DATA:-}" ]]; then rs_probe_data_args=(--set "data.train=$TRAIN_DATA"); fi
  for rs_seed in "${RS_SEEDS[@]}"; do
    rs_dir="$RS_RUN_ROOT/$rs_label/seed_$rs_seed"
    rs_checkpoint="${CHECKPOINT:-$RS_RUN_ROOT/a3_full/seed_$rs_seed/training/checkpoint.pt}"
    rs_probe_calibration_args=()
    rs_calibration="${CALIBRATION_FILE:-}"
    if [[ -z "$rs_calibration" && -z "${CHECKPOINT:-}" && -f "$RS_RUN_ROOT/a3_full/seed_$rs_seed/calibration/calibration.json" ]]; then
      rs_calibration="$RS_RUN_ROOT/a3_full/seed_$rs_seed/calibration/calibration.json"
    fi
    if [[ -n "$rs_calibration" ]]; then
      rs_probe_calibration_args=(--set "controller.calibration_file=$rs_calibration")
    fi
    "$RS_PYTHON" -m repairscope probe --config "$RS_CONFIG" --checkpoint "$rs_checkpoint" \
      --seed "$rs_seed" ${rs_plan_args[@]+"${rs_plan_args[@]}"} \
      --set "probing.strategy=$strategy" --output-dir "$rs_dir" \
      ${rs_probe_data_args[@]+"${rs_probe_data_args[@]}"} \
      ${rs_probe_calibration_args[@]+"${rs_probe_calibration_args[@]}"} \
      ${RS_DRY_ARGS[@]+"${RS_DRY_ARGS[@]}"} "$@"
  done
}
