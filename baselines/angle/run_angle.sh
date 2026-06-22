#!/usr/bin/env bash
# Wait for Word2Vec to finish, then launch ANGLE training on GPU 3.
set -e

W2V_BIN=~/angle_devign/w2v_model.bin
LOG=~/angle_devign/train.log

echo "$(date): Waiting for Word2Vec to finish..." | tee "$LOG"
while [ ! -f "$W2V_BIN" ]; do
    sleep 30
done
echo "$(date): w2v_model.bin found — starting ANGLE training" | tee -a "$LOG"

cd ~/angle_devign
CUDA_VISIBLE_DEVICES=3 ~/venvs/reveal310/bin/python3 train.py 2>&1 | tee -a "$LOG"
echo "$(date): ANGLE training complete." | tee -a "$LOG"
