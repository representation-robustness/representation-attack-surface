#!/bin/bash
# After VulGNN-WithId seed=42 finishes (cache computed), run remaining 4 seeds.
# Seeds 1337/7/100/999 launched on GPUs 1-4 (freed after REVEAL-NoId completes).
# The CodeBERT embeddings cache from seed=42 is reused.
PYTHON=/home/jesse/venvs/reveal310/bin/python
LOGS=/home/jesse/thesis/experiments/logs
EXPDIR=/home/jesse/thesis/experiments
CACHE=/home/jesse/thesis/devign_full/codebert_devign_embs
RESULT=/home/jesse/thesis/devign_full/vulgnn_withid_seed42_results.json

echo "[chain_withid] Waiting for VulGNN-WithId seed=42 to finish..."
until [ -f "$RESULT" ]; do sleep 60; done
echo "[chain_withid] seed=42 done. Launching remaining seeds..."

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $EXPDIR/exp12_vulgnn_withid.py --seed 1337 > $LOGS/exp12_seed1337.log 2>&1 &
echo "[chain_withid] seed=1337 PID=$!"

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $EXPDIR/exp12_vulgnn_withid.py --seed 7    > $LOGS/exp12_seed7.log    2>&1 &
echo "[chain_withid] seed=7    PID=$!"

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON $EXPDIR/exp12_vulgnn_withid.py --seed 100  > $LOGS/exp12_seed100.log  2>&1 &
echo "[chain_withid] seed=100  PID=$!"

CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $EXPDIR/exp12_vulgnn_withid.py --seed 999  > $LOGS/exp12_seed999.log  2>&1 &
echo "[chain_withid] seed=999  PID=$!"

wait
echo "[chain_withid] All VulGNN-WithId seeds complete."
