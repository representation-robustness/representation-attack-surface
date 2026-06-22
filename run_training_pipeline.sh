#!/usr/bin/env bash
# Full autonomous training pipeline.
# Step 1: 300-step GPU probe — checks if weighted BCE fixes convergence.
# Step 2: Full training (16700 steps / 100 epochs max) if probe succeeds.
# Step 3: Fallback to REVEAL GGNN if Devign fails to converge.
# Logs everything to devign_full/pipeline.log

set -euo pipefail

GPU=6
THESIS=/home/jesse/thesis
DEVIGN=$THESIS/baselines/devign
DEVIGN_INPUT=$THESIS/devign_full/devign_input/originals_train
LOG=$THESIS/devign_full/pipeline.log
VENV_DEVIGN=/home/jesse/code/thesis/.venvs/devign
REVEAL_RL=$THESIS/data/raw/ReVeal/Vuld_SySe/representation_learning
VENV_REVEAL=/home/jesse/venvs/reveal310

ts() { date '+[%Y-%m-%d %H:%M:%S]'; }

log() { echo "$(ts) $*" | tee -a "$LOG"; }

log "=== Pipeline started on GPU $GPU ==="
log "THESIS=$THESIS"

# ── Activate devign venv ──────────────────────────────────────────────────────
source "$VENV_DEVIGN/bin/activate"
cd "$DEVIGN"

# ── Step 1: 300-step probe ────────────────────────────────────────────────────
log "STEP 1: 300-step convergence probe"
PROBE_LOG=$THESIS/devign_full/probe_train.log

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --dataset devign_full_probe \
    --input_dir "$DEVIGN_INPUT" \
    --num_epochs 300 \
    --dev_every 167 \
    --log_every 25 \
    --max_patience 5 \
    --lr 1e-4 \
    2>&1 | tee "$PROBE_LOG"

# Check whether loss dropped meaningfully (below 0.60 at any point in probe log)
MIN_LOSS=$(grep -oP 'Train Loss \K[0-9.]+' "$PROBE_LOG" | sort -n | head -1)
log "Probe complete. Minimum train loss observed: $MIN_LOSS"

CONVERGING=0
# Use python for float comparison
python3 -c "import sys; sys.exit(0 if float('$MIN_LOSS') < 0.62 else 1)" && CONVERGING=1 || true

if [ "$CONVERGING" -eq 1 ]; then
    log "✓ PROBE PASSED (loss < 0.62). Proceeding to full training."

    # ── Step 2: Full training ─────────────────────────────────────────────────
    log "STEP 2: Full training (max 16700 steps = 100 epochs, patience 20 epochs)"
    FULL_LOG=$THESIS/devign_full/devign_full_v2_train.log

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset devign_full_originals_v2 \
        --input_dir "$DEVIGN_INPUT" \
        --num_epochs 16700 \
        --dev_every 167 \
        --log_every 50 \
        --max_patience 20 \
        --lr 1e-4 \
        2>&1 | tee "$FULL_LOG"

    # Get best val F1
    BEST_F1=$(grep -oP 'New best model saved\tVal F1 \K[0-9.]+' "$FULL_LOG" | tail -1)
    log "Full training complete. Best Val F1: $BEST_F1"

    # Check if model learned (F1 > 65 means genuinely learning)
    LEARNED=0
    python3 -c "import sys; sys.exit(0 if float('${BEST_F1:-0}') > 65.0 else 1)" && LEARNED=1 || true

    if [ "$LEARNED" -eq 1 ]; then
        log "✓ MODEL CONVERGED (Val F1 > 65%). Extracting embeddings."

        # ── Step 2b: Extract embeddings ──────────────────────────────────────
        log "Extracting GGNN embeddings from v2 model..."
        # Update extraction script to use v2 model
        CHECKPOINT=$DEVIGN/models/devign_full_originals_v2/best_model.pt
        python extract_ggnn_embeddings.py --checkpoint "$CHECKPOINT" \
            2>&1 | tee "$THESIS/devign_full/ggnn_extraction_v2.log"

        log "Running robustness eval..."
        source "$VENV_REVEAL/bin/activate"
        cd "$REVEAL_RL"
        python -u reveal_robustness_eval.py \
            --output_json robustness_results_v2.json \
            2>&1 | tee "$THESIS/devign_full/robustness_eval_v2.log"

        log "=== PIPELINE COMPLETE — results in robustness_eval_v2.log ==="
        exit 0
    else
        log "✗ Model F1=$BEST_F1 — still degenerate despite weighted loss. Falling through to Step 3."
    fi
else
    log "✗ PROBE FAILED (min loss $MIN_LOSS >= 0.62). Skipping full training. Going to Step 3."
fi

# ── Step 3: Fallback — train REVEAL's own GGNN ───────────────────────────────
log "STEP 3: Fallback — training REVEAL GGNN (gnn.py, pure PyTorch, no DGL)"
source "$VENV_REVEAL/bin/activate"

REVEAL_TRAIN=$THESIS/data/raw/ReVeal/data_processing
cd "$THESIS"

python data/raw/ReVeal/data_processing/build_obf_ggnn_only.py \
    2>&1 | tee "$THESIS/devign_full/reveal_ggnn_train.log"

log "=== Step 3 complete. Check reveal_ggnn_train.log ==="
