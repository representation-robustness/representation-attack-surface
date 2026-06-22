#!/bin/bash
# After exp7_reveal_aug_prep.py finishes (GGNN inputs built),
# launch REVEAL-Aug for all 5 seeds on GPUs 1-5.
# (Waits for REVEAL-NoId to finish first if needed, since both use GPUs 1-5)
PYTHON=/home/jesse/venvs/reveal310/bin/python
LOGS=/home/jesse/thesis/experiments/logs
EXPDIR=/home/jesse/thesis/experiments
READY=/home/jesse/thesis/devign_full/devign_input/obf_controlflow_train/train_GGNNinput.json
NOID_DONE=/home/jesse/thesis/devign_full/reveal_noid_seed999_results.json

echo "[chain_reveal_aug] Waiting for prep data (obf_controlflow_train GGNN JSON)..."
until [ -f "$READY" ]; do sleep 60; done
echo "[chain_reveal_aug] Prep data ready."

echo "[chain_reveal_aug] Waiting for REVEAL-NoId to free GPUs 1-5..."
until [ -f "$NOID_DONE" ]; do sleep 120; done
echo "[chain_reveal_aug] REVEAL-NoId done. Launching REVEAL-Aug 5 seeds..."

SEEDS=(42 1337 7 100 999)
GPUS=(1 2 3 4 5)
for i in 0 1 2 3 4; do
    SEED=${SEEDS[$i]}
    GPU=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON $EXPDIR/exp7_reveal_aug_train.py --seed $SEED \
        > $LOGS/exp7_aug_seed${SEED}.log 2>&1 &
    echo "[chain_reveal_aug] seed=$SEED PID=$! GPU=$GPU"
done

wait
echo "[chain_reveal_aug] All REVEAL-Aug seeds complete."
