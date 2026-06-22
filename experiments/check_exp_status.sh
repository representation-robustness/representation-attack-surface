#!/bin/bash
# Quick status check for all running experiments
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=== REVEAL-NoId (exp10) ==="
for SEED in 42 1337 7 100 999; do
    LOG=/home/jesse/thesis/experiments/logs/exp10_seed${SEED}.log
    if [ -f "/home/jesse/thesis/devign_full/reveal_noid_seed${SEED}_results.json" ]; then
        F1=$(python3 -c "import json; d=json.load(open('/home/jesse/thesis/devign_full/reveal_noid_seed${SEED}_results.json')); print(f\"done F1={d['clean']['f1']:.2f}\")")
        echo "  seed=$SEED $F1"
    elif [ -f "$LOG" ]; then
        LAST=$(tail -1 $LOG)
        echo "  seed=$SEED running: $LAST"
    else
        echo "  seed=$SEED not started"
    fi
done

echo ""
echo "=== VulGNN-WithId (exp12) ==="
LOG=/home/jesse/thesis/experiments/logs/exp12_seed42.log
if [ -f "/home/jesse/thesis/devign_full/vulgnn_withid_seed42_results.json" ]; then
    echo "  seed=42 DONE"
else
    tail -2 $LOG 2>/dev/null | sed 's/^/  /'
fi

echo ""
echo "=== REVEAL-Aug Prep (exp7) ==="
cat /home/jesse/thesis/experiments/logs/exp7_prep.log | grep -E "batch|Moved|GGNN|complete|Error" | tail -5 | sed 's/^/  /'
PARSED_DEAD=$(ls /home/jesse/thesis/devign_full/devign_input/parsed_cache/obf_deadcode/ 2>/dev/null | wc -l)
PARSED_CF=$(ls /home/jesse/thesis/devign_full/devign_input/parsed_cache/obf_controlflow/ 2>/dev/null | wc -l)
echo "  parsed: obf_deadcode=$PARSED_DEAD, obf_controlflow=$PARSED_CF"

echo ""
echo "=== Processes ==="
ps aux | grep "exp7\|exp10\|exp12\|chain" | grep python | grep -v grep | awk '{print $1,$11,$12}' | head -10
