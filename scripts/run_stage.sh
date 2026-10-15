#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RS=.venv/bin/repairscope
PY=.venv/bin/python
CFG=runs/configs/appworld.json
test -f "$CFG" || { echo 'Prepare pinned models/configs first.' >&2; exit 1; }
case "${1:-}" in
  pilot)
    "$RS" run --config "$CFG" --split train --method baseline --experiment pilot_baseline_train --limit 3
    "$RS" run --config "$CFG" --split train --method capture --experiment pilot_capture_train --limit 3
    "$RS" run --config "$CFG" --split dev --method capture --experiment pilot_capture_dev --limit 3
    ;;
  capture)
    for split in train dev; do
      "$RS" run --config "$CFG" --split "$split" --method capture --experiment "capture_$split"
    done
    ;;
  labels)
    for split in train dev; do
      "$RS" map-goals --config "$CFG" --trajectories "runs/capture_$split" --output "runs/labels/$split.mapping.json"
      "$RS" collect --config "$CFG" --trajectories "runs/capture_$split" --mapping "runs/labels/$split.mapping.json" --output "runs/labels/$split.jsonl"
      "$RS" quality --data "runs/labels/$split.jsonl" --output "runs/labels/$split.quality.json"
    done
    echo 'Audit goal mappings and write runs/labels/mapping_audit.json using docs/LABEL_AUDIT.md.'
    ;;
  folds)
    "$PY" scripts/check_collection.py
    "$PY" scripts/prepare_folds.py
    for kind in effect scalar; do
      for fold in 0 1 2 3 4; do
        "$RS" train --config "runs/configs/train_$kind.json" --fold "$fold" --output-dir "runs/checkpoints/${kind}_fold$fold"
        "$RS" evaluate-predictor --checkpoint "runs/checkpoints/${kind}_fold$fold/best" --data "runs/folds/fold$fold.validation.jsonl" --output "runs/diagnostics/${kind}_fold$fold.json"
      done
    done
    ;;
  train)
    "$PY" scripts/check_collection.py
    for kind in effect scalar; do
      for seed in 100 101 102; do
        "$RS" train --config "runs/configs/train_$kind.json" --seed "$seed" --output-dir "runs/checkpoints/${kind}_s$seed"
        "$RS" evaluate-predictor --checkpoint "runs/checkpoints/${kind}_s$seed/best" --data runs/labels/dev.jsonl --output "runs/diagnostics/${kind}_s${seed}_dev.json"
      done
    done
    ;;
  dev|test)
    phase="$1"
    if [[ "$phase" == test ]]; then
      "$PY" scripts/freeze_protocol.py verify
      splits=(test_normal test_challenge)
      extra=(--final)
    else
      splits=(dev)
      extra=()
    fi
    for split in "${splits[@]}"; do
      for method in baseline direct; do
        experiment="${method}_${split}_s100"
        "$RS" run --config "$CFG" --split "$split" --method "$method" --experiment "$experiment" "${extra[@]}"
        "$RS" evaluate-native --config "$CFG" --experiment "$experiment"
      done
      for method in effect scalar; do
        for seed in 100 101 102; do
          experiment="${method}_${split}_s$seed"
          "$RS" run --config "$CFG" --split "$split" --method "$method" --checkpoint "runs/checkpoints/${method}_s$seed/best" --experiment "$experiment" "${extra[@]}"
          "$RS" evaluate-native --config "$CFG" --experiment "$experiment"
        done
      done
    done
    ;;
  *) echo 'Usage: scripts/run_stage.sh {pilot|capture|labels|folds|train|dev|test}' >&2; exit 2 ;;
esac
