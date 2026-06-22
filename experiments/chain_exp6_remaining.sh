#!/bin/bash
# Waits for exp3+4 (inference) to finish freeing GPUs 1+2, then runs exp6 seeds 100 and 999
PYTHON=/home/jesse/venvs/reveal310/bin/python
LOGS=/home/jesse/thesis/experiments/logs
EXPDIR=/home/jesse/thesis/experiments
PREDS=/home/jesse/thesis/devign_full/attack/preds

echo "[chain6] Waiting for preds files before claiming GPUs 1+2 for exp6 seeds 100,999..."

wait_for_output() {
    local file=$1
    while [ ! -f "$file" ]; do
        sleep 30
    done
    echo "[chain6] $file found!"
}

wait_for_output "$PREDS/codebert_preds.json"
wait_for_output "$PREDS/regvd_preds.json"

echo "[chain6] Inference complete. Launching Exp6 seeds 100 + 999..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $EXPDIR/exp6_regvd_aug_train.py --seed 100 > $LOGS/exp6_seed100.log 2>&1 &
echo "[chain6] exp6 seed100 PID: $!"
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $EXPDIR/exp6_regvd_aug_train.py --seed 999 > $LOGS/exp6_seed999.log 2>&1 &
echo "[chain6] exp6 seed999 PID: $!"
wait
echo "[chain6] Exp6 seeds 100+999 complete."
