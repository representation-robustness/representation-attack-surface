#!/usr/bin/env bash
# Run after devign_full/*.c trees and originals_manifest.csv exist.
# Full slicing is long (~27k × 4 Joern runs); use nohup or tmux on the server.
set -euo pipefail
THESIS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
SLICE="${THESIS_ROOT}/data/raw/ReVeal/data_processing/make_devign_full_slices.py"
PKG="${THESIS_ROOT}/data/raw/ReVeal/data_processing/build_devign_full_full_data_with_slices.py"
GGNN="${THESIS_ROOT}/data/raw/ReVeal/data_processing/build_devign_full_split_and_ggnn.py"

echo "=== 1/3 Slicing (resume-safe) ==="
"$PY" -u "$SLICE" --variants originals obf_identifier obf_deadcode obf_controlflow

echo "=== 2/3 Package aligned JSON with slices ==="
"$PY" -u "$PKG"

echo "=== 3/3 Split 80/10/10 + GGNN inputs ==="
"$PY" -u "$GGNN"

echo "Done. Check devign_full/*_slicing_log.json and devign_full/devign_full_split_and_ggnn_summary.json"
