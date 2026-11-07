#!/usr/bin/env bash
# One-screen progress overview of the v2 rerun. Read-only; safe to run any time.
R=${RERUN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/rerun}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${RERUN_PY:-$R/venvs/main/bin/python}
echo "== GPU queue (daemon pid $(cat $R/runs/gpu_queue.pid 2>/dev/null), alive: $(ps -p "$(cat $R/runs/gpu_queue.pid 2>/dev/null)" >/dev/null 2>&1 && echo yes || echo NO))"
(cd "$REPO" && $PY -m pipeline.common.gpu_queue status | cut -c1-150)
echo "   last events:"; tail -3 $R/logs/gpu_queue.log | sed 's/^/   /'
echo "== GPU guard"
(cd "$REPO" && $PY -m pipeline.common.gpu_guard status)
echo "== Graph builds"
cat $R/logs/build_all_graphs.log 2>/dev/null | sed 's/^/   /'
cat $R/logs/cpu_after_graphs.log 2>/dev/null | grep -E "rc=|clean F1" | sed 's/^/   /'
echo "== Prediction files"
for d in $R/results/preds/*/*/; do printf "   %-12s %-22s %s runs\n" "$(basename $(dirname $d))" "$(basename $d)" "$(ls $d | wc -l)"; done 2>/dev/null
echo "== Disk: $(du -sh $R 2>/dev/null | cut -f1) used by rerun; /home free: $(df -h /home | awk 'NR==2{print $4}')"
echo "== My CPU now: $(top -b -n 2 -d 2 | awk '/^ *PID/{b++} b==2 && $2=="'"$USER"'"{c+=$9} END{printf "%.0f%%", c}')"
