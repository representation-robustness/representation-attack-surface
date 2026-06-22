#!/bin/bash
# Waits for exp3 (CodeBERT) to finish, then:
#   1) runs exp5 (transfer matrix, CPU)
#   2) launches exp8 (CodeBERT-Mixed) on GPUs 1+2+6 (2 at a time, 5 seeds total)
#   3) launches exp6 seeds 100+999 on GPUs 1+2 after exp8 seeds 42+1337 finish
#
# Exp6 seeds 42/1337/7 are running on GPUs 3/4/5 (training jobs, ~hours)
# Using GPU6 (always free) avoids any conflict with those.

PYTHON=/home/jesse/venvs/reveal310/bin/python
LOGS=/home/jesse/thesis/experiments/logs
EXPDIR=/home/jesse/thesis/experiments
PREDS=/home/jesse/thesis/devign_full/attack/preds

echo "[chain] Started. Waiting for codebert_preds.json..."

# Wait for exp3 (regvd_preds.json already exists)
while [ ! -f "$PREDS/codebert_preds.json" ]; do sleep 30; done
echo "[chain] codebert_preds.json found — running Exp5 transfer matrix (CPU)..."

$PYTHON $EXPDIR/exp5_transfer_matrix.py > $LOGS/exp5.log 2>&1
echo "[chain] Exp5 done (exit=$?). Output: $PREDS/../transfer_asr_matrix.json"

echo "[chain] Launching Exp8 batch 1: seeds 42+1337 on GPUs 1+2..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $EXPDIR/exp8_codebert_mixed_train.py --seed 42   > $LOGS/exp8_seed42.log   2>&1 &
PID42=$!
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $EXPDIR/exp8_codebert_mixed_train.py --seed 1337 > $LOGS/exp8_seed1337.log 2>&1 &
PID1337=$!
echo "[chain]   seed42 PID=$PID42  seed1337 PID=$PID1337"

echo "[chain] Launching Exp8 batch 2: seed 7 on GPU6 (runs in parallel with batch 1)..."
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON $EXPDIR/exp8_codebert_mixed_train.py --seed 7    > $LOGS/exp8_seed7.log   2>&1 &
PID7=$!
echo "[chain]   seed7 PID=$PID7"

echo "[chain] Waiting for seeds 42+1337+7 to finish..."
wait $PID42 $PID1337 $PID7
echo "[chain] Batch 1+2 done."

echo "[chain] Launching Exp8 batch 3: seeds 100+999 on GPUs 1+2..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $EXPDIR/exp8_codebert_mixed_train.py --seed 100  > $LOGS/exp8_seed100.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $EXPDIR/exp8_codebert_mixed_train.py --seed 999  > $LOGS/exp8_seed999.log  2>&1 &

echo "[chain] Launching Exp6 seeds 100+999 on GPUs 6+... (will wait for GPU6 to free up)"
# GPU6 is free after seed7 finishes — but batch3 uses 1+2, so GPU6 is available now
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON $EXPDIR/exp6_regvd_aug_train.py --seed 100 > $LOGS/exp6_seed100.log 2>&1 &
echo "[chain]   exp6 seed100 PID=$!"

wait
echo "[chain] Exp8 seeds 100+999 done. Launching exp6 seed 999..."
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $EXPDIR/exp6_regvd_aug_train.py --seed 999 > $LOGS/exp6_seed999.log 2>&1 &
echo "[chain]   exp6 seed999 PID=$!"
wait
echo "[chain] All done."
