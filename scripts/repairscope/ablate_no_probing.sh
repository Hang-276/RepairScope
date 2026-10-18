#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
RS_PROBE_LABEL=ablate_no_probing rs_probe risk --variant no_probing "$@"
