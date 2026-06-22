#!/usr/bin/env bash
# Full obfuscation evaluation pipeline — runs all 4 steps sequentially.
# Expected runtime: ~67h slicing + ~30min GGNN + ~5min eval.
# Run with: nohup bash devign_full/run_obf_pipeline.sh > devign_full/obf_pipeline.log 2>&1 &

set -euo pipefail

# Joern JARs compiled at class file 56.0 (Java 12+) — must use local JDK 17
export JAVA_HOME=/home/jesse/local/jdk-17.0.18+8
export PATH=$JAVA_HOME/bin:/home/jesse/venvs/reveal310/bin:$PATH

THESIS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/jesse/venvs/reveal310/bin/python"

DATA_PROC="$THESIS_ROOT/data/raw/ReVeal/data_processing"
DEVIGN="$THESIS_ROOT/baselines/devign"
DEVIGN_FULL="$THESIS_ROOT/devign_full"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Slicing ──────────────────────────────────────────────────────────
log "=== STEP 1/4: SLICING (obf_identifier, obf_deadcode, obf_controlflow) ==="
log "Resume-safe — already-sliced files will be skipped."
"$PY" -u "$DATA_PROC/make_devign_full_slices.py" \
    --variants obf_identifier obf_deadcode obf_controlflow
log "=== STEP 1 DONE ==="

# ── Step 2: Packaging ────────────────────────────────────────────────────────
log "=== STEP 2/4: PACKAGING (aligning all variants) ==="
"$PY" -u "$DATA_PROC/build_devign_full_full_data_with_slices.py"
log "=== STEP 2 DONE ==="

# ── Step 3: GGNN inputs (obf test sets only) ─────────────────────────────────
log "=== STEP 3/4: GGNN BUILD (obf test sets only — skipping originals) ==="
"$PY" -u "$DATA_PROC/build_obf_ggnn_only.py"
log "=== STEP 3 DONE ==="

# ── Step 4: Evaluate controlflow against best_model.pt ───────────────────────
log "=== STEP 4/4: EVALUATION — obf_controlflow vs best_model.pt ==="
cd "$DEVIGN"
CUDA_VISIBLE_DEVICES=0 "$PY" -u main.py \
    --model_type devign \
    --dataset devign_full_originals \
    --input_dir "$DEVIGN_FULL/devign_input/obf_controlflow_test" \
    --feature_size 169 \
    --graph_embed_size 200 \
    --num_steps 6 \
    --batch_size 128 \
    --checkpoint "$DEVIGN/models/devign_full_originals/best_model.pt" \
    --eval_only
log "=== STEP 4 DONE — PIPELINE COMPLETE ==="
