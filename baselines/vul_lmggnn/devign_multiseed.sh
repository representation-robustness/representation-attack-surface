#!/usr/bin/env bash
# Run LMGGNN training with 5 random seeds and aggregate results.
# Each seed: ~4 hours. Total: ~20 hours (overnight).
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_multiseed.sh
set -e

PYTHON=~/venvs/reveal310/bin/python3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=~/thesis/devign_full
SEEDS=(42 1337 7 100 999)

echo "=== LMGGNN Multi-seed Training ===" | tee "$LOG_DIR/lmggnn_multiseed.log"
echo "Started: $(date)" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_DIR/lmggnn_multiseed.log"

for SEED in "${SEEDS[@]}"; do
    echo "" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
    echo "======================================" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
    echo "  Seed $SEED  ($(date))" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
    echo "======================================" | tee -a "$LOG_DIR/lmggnn_multiseed.log"

    $PYTHON "$SCRIPT_DIR/train_devign.py" \
        --train --eval --epochs 10 \
        --seed "$SEED" \
        2>&1 | tee -a "$LOG_DIR/lmggnn_multiseed.log"

    echo "  Seed $SEED done: $(date)" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
done

echo "" | tee -a "$LOG_DIR/lmggnn_multiseed.log"
echo "All seeds done: $(date)" | tee -a "$LOG_DIR/lmggnn_multiseed.log"

# Aggregate results
$PYTHON - <<'EOF' | tee -a "$LOG_DIR/lmggnn_multiseed.log"
import json, os, numpy as np
from pathlib import Path

result_dir = Path.home() / "thesis/devign_full"
seeds = [42, 1337, 7, 100, 999]
conditions = ["original", "identifier", "deadcode", "controlflow"]

all_results = []
for seed in seeds:
    p = result_dir / f"lmggnn_seed{seed}_results.json"
    if p.exists():
        all_results.append(json.load(open(p)))
    else:
        print(f"  WARN: {p} not found")

if not all_results:
    print("No results to aggregate.")
    exit(1)

agg = {"n_seeds": len(all_results), "seeds": seeds, "model": "LMGGNN"}
for cond in conditions:
    f1s  = [r[cond]["f1"]  for r in all_results if cond in r]
    accs = [r[cond]["acc"] for r in all_results if cond in r]
    prs  = [r[cond]["pr"]  for r in all_results if cond in r]
    rcs  = [r[cond]["rc"]  for r in all_results if cond in r]
    agg[cond] = {
        "f1_mean": round(float(np.mean(f1s)), 2),
        "f1_std":  round(float(np.std(f1s)), 2),
        "acc_mean": round(float(np.mean(accs)), 2),
        "pr_mean":  round(float(np.mean(prs)), 2),
        "rc_mean":  round(float(np.mean(rcs)), 2),
        "all_f1": f1s,
    }

base = agg["original"]["f1_mean"]
for cond in conditions[1:]:
    agg[cond]["delta_f1"] = round(agg[cond]["f1_mean"] - base, 2)

out = result_dir / "lmggnn_multiseed_results.json"
with open(out, "w") as f:
    json.dump(agg, f, indent=2)

print("\n=== LMGGNN Multi-seed Summary ===")
for cond in conditions:
    m = agg[cond]
    delta = f"  ΔF1={m.get('delta_f1',0):+.2f}" if cond != "original" else ""
    print(f"  {cond:<20} F1={m['f1_mean']:.2f} ± {m['f1_std']:.2f}%{delta}")
print(f"\nSaved → {out}")
EOF
