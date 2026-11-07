#!/usr/bin/env bash
# Every stage, start to finish, unattended: the driver used for the 2026-11-07 independent rerun, with its paths
# made configurable. GPU jobs are queued first and start as soon as their data dependencies exist, while the CPU
# stages (data, transforms, graphs, CPU baselines, DeepWukong) run on 4 pinned cores.
#
#   RERUN_ROOT=/path/new_root DATASET_RELEASE=/path/dataset_release [REF=/path/earlier_root] tools/full_rerun.example.sh
#
# Settings used in that rerun. The driver started the queue with 2 jobs/GPU and 6 slots; the queue daemon was then
# restarted (running jobs adopted) with the values below, which were in effect for almost all of the run.
# GPUs were further restricted live with $RERUN_ROOT/runs/gpu_allowed ("3,4,5,6").
set -uo pipefail
cd "$(dirname "$0")/.."
export RERUN_ROOT=${RERUN_ROOT:?set RERUN_ROOT} DATASET_RELEASE=${DATASET_RELEASE:?set DATASET_RELEASE}
export RERUN_CPUS=28-31 RERUN_THREADS=4                         # CPU stages: 4 cores
export RERUN_MAX_GPUS=4 RERUN_JOBS_PER_GPU=4 RERUN_SLOT_THREADS=2  # <= 4 GPUs, <= 4 of our jobs per GPU
export RERUN_SLOT_CPUS="12-13 14-15 16-17 18-19 20-21 22-23 24-25 26-27"   # 8 queue slots x 2 cores
[[ -n ${REF:-} ]] && export RERUN_PRIORITY_FROM=$REF            # longest jobs first, by REF's durations
log() { echo "[$(date -Is)] $*"; }
log start; ./reproduce.sh check
log "enqueue GPU jobs + start queue (jobs wait for their data dependencies)"; ./reproduce.sh gpu_models
for s in fetch data transforms graphs cpu_models; do log "stage $s"; ./reproduce.sh $s || { log "stage $s FAILED"; exit 1; }; done
log "stage deepwukong"; RERUN_GPU_JOB_CPUS=28-30 ./reproduce.sh deepwukong || log "deepwukong FAILED"
log "waiting for the GPU queue to drain"
Q=$RERUN_ROOT/runs/queue
while [[ -n $(find "$Q/pending" "$Q/running" -name '*.json' 2>/dev/null) ]]; do sleep 900; done
log "queue drained: $(ls "$Q/done" | wc -l) done, $(ls "$Q/failed" | wc -l) failed"
./reproduce.sh results
PY=${RERUN_PY:-$RERUN_ROOT/venvs/main/bin/python}
if [[ -n ${REF:-} ]]; then   # reproduction check against the earlier run
  nice -n 19 taskset -c "$RERUN_CPUS" $PY -m pipeline.report.compare --ref "$REF" --new "$RERUN_ROOT" > "$RERUN_ROOT/results/compare_artifacts.txt" 2>&1
  nice -n 19 $PY -m pipeline.report.compare_findings --ref "$REF" --new "$RERUN_ROOT"
fi
nice -n 19 $PY -m pipeline.report.gpu_usage --root "$RERUN_ROOT" ${REF:+--ref "$REF"} > /dev/null
log finished
