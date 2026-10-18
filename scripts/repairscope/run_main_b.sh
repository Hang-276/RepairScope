#!/usr/bin/env bash
set -euo pipefail
RS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for rs_strategy in random uncertainty novelty_learnability risk; do
  bash "$RS_SCRIPT_DIR/probe_$rs_strategy.sh" "$@"
done
