#!/usr/bin/env bash
# Eval LMGGNN on all 7 obfuscation conditions for Devign (no retraining).
# Uses existing checkpoints: lmggnn_seed{N}.pt

PYTHON=~/venvs/reveal310/bin/python3
CUDA=5
LOG=~/thesis/devign_full/lmggnn_7cond.log

echo "Starting LMGGNN 7-cond eval" | tee $LOG
date | tee -a $LOG

cd ~/vul-LMGGNN

for SEED in 42 1337 7 100 999; do
    CKPT=~/vul-LMGGNN/data/model/lmggnn_seed${SEED}.pt
    if [ ! -f "$CKPT" ]; then
        echo "SKIP seed $SEED: checkpoint not found" | tee -a $LOG
        continue
    fi
    echo "" | tee -a $LOG
    echo "=== Seed $SEED ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$CUDA $PYTHON train_devign.py \
        --eval \
        --seed $SEED \
        --run_id "seed${SEED}" \
        2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo "Done" | tee -a $LOG
date | tee -a $LOG
