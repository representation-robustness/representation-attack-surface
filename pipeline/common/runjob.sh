#!/usr/bin/env bash
# Run one pipeline job politely on the shared machine and record provenance.
#
#   pipeline/common/runjob.sh [--gpus N] JOB_NAME -- cmd args...
#
# - CPU: nice 19 + idle I/O class; BLAS/torch thread pools capped at $RERUN_THREADS (default 8);
#   optional hard cap: RERUN_CPUS="28-31" pins the job (and all its children) to those cores.
# - GPU (--gpus N): acquire via gpu_guard (free = no other user's process); exit 75 if none
#   so the caller can defer. The lock is released on exit.
# - Writes $RERUN_ROOT/logs/JOB_NAME.log and $RERUN_ROOT/runs/manifests/JOB_NAME.json.
set -euo pipefail

RERUN_ROOT=${RERUN_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/rerun}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# Tool locations default to the reference host layout; the container overrides them (env/install_env.sh).
PY=${RERUN_PY:-$RERUN_ROOT/venvs/main/bin/python}
export RERUN_ROOT
export JAVA_HOME=${RERUN_JDK:-$RERUN_ROOT/tools/jdk}
export PATH=$JAVA_HOME/bin:$PATH
THREADS=${RERUN_THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$RERUN_ROOT/cache/hf
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

NGPU=0
if [[ "${1:-}" == "--gpus" ]]; then NGPU=$2; shift 2; fi
JOB=$1; shift
[[ "${1:-}" == "--" ]] && shift

mkdir -p "$RERUN_ROOT/logs" "$RERUN_ROOT/runs/manifests"
LOG=$RERUN_ROOT/logs/$JOB.log
MAN=$RERUN_ROOT/runs/manifests/$JOB.json

GPUS=""
if (( NGPU > 0 )); then
  EXCL=(); [[ "${RERUN_EXCLUSIVE:-0}" == 1 ]] && EXCL=(--exclusive)
  if ! GOT=$(cd "$REPO" && $PY -m pipeline.common.gpu_guard acquire --n "$NGPU" --job "$JOB" --pid $$ "${EXCL[@]}"); then
    echo "[$(date -Is)] $JOB deferred: no free GPU" | tee -a "$RERUN_ROOT/logs/deferred.log"
    exit 75
  fi
  trap '(cd "$REPO" && $PY -m pipeline.common.gpu_guard release --job "$JOB")' EXIT
  GPUS=${GOT%%;*}
  # shared idle GPU or packed GPU: cap our memory (see gpu_guard)
  [[ $GOT == *";"* ]] && export RERUN_GPU_MEM_MIB=${GOT##*;}
  export CUDA_VISIBLE_DEVICES=$GPUS
else
  export CUDA_VISIBLE_DEVICES=""
fi

START=$(date -Is)
set +e
PIN=()
[[ -n "${RERUN_CPUS:-}" ]] && PIN=(taskset -c "$RERUN_CPUS")
"${PIN[@]}" nice -n 19 ionice -c3 "$@" >"$LOG" 2>&1
RC=$?
set -e

$PY - "$MAN" <<EOF
import json, os, sys, hashlib, subprocess
freeze = subprocess.run(["$PY", "-m", "pip", "freeze"], capture_output=True).stdout
git = lambda *a: subprocess.run(["git", "-C", "$REPO", *a], capture_output=True, text=True).stdout.strip()
json.dump({
    "job": "$JOB", "cmd": $(printf '%s\n' "$@" | $PY -c 'import json,sys;print(json.dumps(sys.stdin.read().splitlines()))'),
    "start": "$START", "end": "$(date -Is)", "exit_code": $RC,
    "gpus": "$GPUS", "gpu_mem_cap_mib": "${RERUN_GPU_MEM_MIB:-}", "threads": $THREADS, "host": "$(hostname)",
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),  # tracked files only
    "git_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
    "env_freeze_sha256": hashlib.sha256(freeze).hexdigest(), "log": "$LOG",
    "container": os.environ.get("APPTAINER_CONTAINER") or os.environ.get("RERUN_IMAGE", ""),
}, open(sys.argv[1], "w"), indent=2)
EOF
exit $RC
