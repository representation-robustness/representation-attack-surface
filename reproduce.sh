#!/usr/bin/env bash
# One entry point for the whole v2 rerun. See docs/REPRODUCE.md.
#
#   ./reproduce.sh [--sif IMAGE.sif] STAGE
#
# STAGE (run in this order; each is resumable, finished jobs are skipped):
#   check        environment sanity: tools, pinned packages, data inputs
#   test         unit tests (statistics, attack definitions, transform invariants, executable semantic tests)
#   fetch        download the Devign source file (checksum-verified)
#   data         canonical clean splits (needs DATASET_RELEASE for Big-Vul / DiverseVul)
#   transforms   v3 transforms + validity gate + semantic tests + audit sample
#   graphs       Joern graphs for every split/condition (CPU-heavy, capped at 4 cores)
#   cpu_models   TF-IDF+LR and CPG+LR (30 runs each)
#   smoke        ~10-min CPU check that every neural model trains and predicts (needs data..graphs for Devign)
#   gpu_models   enqueue every neural run and start the polite GPU queue daemon (only free GPUs)
#   results      metrics, statistics, attacks and LaTeX/CSV tables from the saved predictions
#   status       one-screen progress overview
#   deepwukong   DeepWukong case study on SARD CWE-119 (needs data/deepwukong/raw/Data.7z; host only, see docs)
#   verify [REF] ~20-min CPU check: rerun a deterministic slice and compare id-by-id with REF (another RERUN_ROOT)
#   all          check..cpu_models, then gpu_models (results: rerun `results` as runs finish)
#
# Environment: RERUN_ROOT (data/results root, default ../rerun next to the repo), DATASET_RELEASE
# (released Big-Vul/DiverseVul splits), RERUN_CPUS (cores for CPU stages, default the last 4).
# GPU: RERUN_MAX_GPUS (default 2), RERUN_GPUS (allowed ids, e.g. "0,1,2,3"), RERUN_SHARE_IDLE (0 = never
# share a GPU with another user's process). Subset of the GPU matrix: RERUN_SEEDS, RERUN_DATASETS, RERUN_JOBS.
# With --sif, the stage runs inside the Apptainer image (tools come from the image; repo and data are bound).
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
export RERUN_ROOT=${RERUN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/rerun}
export DATASET_RELEASE=${DATASET_RELEASE:-$(cd "$(dirname "$0")/.." && pwd)/dataset_release}

if [[ "${1:-}" == "--sif" ]]; then
  SIF=$2; shift 2
  APPTAINER=$(command -v apptainer || echo "$RERUN_ROOT/tools/apptainer/bin/apptainer")
  NV=(); [[ "${1:-}" == gpu_models || "${1:-}" == all || "${1:-}" == status ]] && NV=(--nv)
  # --contain/--cleanenv isolate the host env; deliberately NOT --pid: the GPU guard must see other users' GPU processes.
  mkdir -p "$RERUN_ROOT/cache/home"
  exec "$APPTAINER" exec "${NV[@]}" --contain --cleanenv --home "$RERUN_ROOT/cache/home" --pwd "$REPO" \
      --bind "$REPO:$REPO,$RERUN_ROOT:$RERUN_ROOT" \
      $( [[ -d $DATASET_RELEASE ]] && echo --bind "$DATASET_RELEASE:$DATASET_RELEASE:ro" ) \
      --env "$(for v in RERUN_ROOT DATASET_RELEASE RERUN_CPUS RERUN_MAX_GPUS RERUN_GPUS RERUN_SHARE_IDLE \
                       RERUN_SLOT_CPUS RERUN_SLOT_THREADS RERUN_JOBS_PER_GPU RERUN_PRIORITY_FROM RERUN_SEEDS RERUN_DATASETS RERUN_JOBS; do
                 [[ -n ${!v:-} ]] && printf '%s=%s,' "$v" "${!v}"; done | sed 's/,$//')" \
      "$SIF" "$REPO/reproduce.sh" "$@"
fi

PY=${RERUN_PY:-$RERUN_ROOT/venvs/main/bin/python}
export RERUN_PY=$PY PYTHONPATH=$REPO
export RERUN_CPUS=${RERUN_CPUS:-$(nproc --all | awk '{print $1-4 "-" $1-1}')}
export RERUN_THREADS=${RERUN_THREADS:-4}
cd "$REPO"
mkdir -p "$RERUN_ROOT/logs" "$RERUN_ROOT/cache/home"

check() {
  echo "python : $($PY --version 2>&1) ($PY)"
  echo "java   : $("${RERUN_JDK:-$RERUN_ROOT/tools/jdk}/bin/java" -version 2>&1 | head -1)"
  echo "joern  : $(ls -d "${RERUN_JOERN:-$RERUN_ROOT/tools/joern-v4.0.655}/joern-cli" 2>/dev/null || echo MISSING)"
  echo "gcc    : $(gcc --version | head -1)"
  if diff -q <($PY -m pip freeze 2>/dev/null) env/requirements-main.lock.txt >/dev/null; then
    echo "pip    : matches env/requirements-main.lock.txt"
  else
    echo "pip    : DIFFERS from env/requirements-main.lock.txt:"; diff <($PY -m pip freeze 2>/dev/null) env/requirements-main.lock.txt | head -20
  fi
  echo "data   : RERUN_ROOT=$RERUN_ROOT DATASET_RELEASE=$DATASET_RELEASE $( [[ -d $DATASET_RELEASE/in_repo ]] && echo '(found)' || echo '(MISSING in_repo/)')"
  echo "cores  : CPU stages pinned to $RERUN_CPUS, $RERUN_THREADS threads"
}

start_queue() {
  local pidf=$RERUN_ROOT/runs/gpu_queue.pid
  if [[ -f $pidf ]] && ps -p "$(cat "$pidf")" >/dev/null 2>&1; then
    echo "GPU queue daemon already running (pid $(cat "$pidf"))"; return
  fi
  nohup setsid nice -n 19 "$PY" -m pipeline.common.gpu_queue run >>"$RERUN_ROOT/logs/gpu_queue.log" 2>&1 &
  echo $! > "$pidf"
  echo "GPU queue daemon started (pid $!); log: $RERUN_ROOT/logs/gpu_queue.log"
}

# Quick (~20 min, CPU only) end-to-end check against a reference run: a small deterministic slice of the
# pipeline is rerun into $RERUN_ROOT/verify and compared id-by-id with REF (default: $RERUN_ROOT itself).
verify() {
  local ref=${1:-$RERUN_ROOT} v=$RERUN_ROOT/verify
  export RERUN_JDK=${RERUN_JDK:-$RERUN_ROOT/tools/jdk} RERUN_JOERN=${RERUN_JOERN:-$RERUN_ROOT/tools/joern-v4.0.655}
  export RERUN_UPSTREAM=${RERUN_UPSTREAM:-$RERUN_ROOT/upstream} HF_HOME=${HF_HOME:-$RERUN_ROOT/cache/hf}
  rm -rf "$v" && mkdir -p "$v/data/devign/raw"
  [[ -f $ref/data/devign/raw/function.json ]] && cp "$ref/data/devign/raw/function.json" "$v/data/devign/raw/"
  local rj=pipeline/common/runjob.sh
  RERUN_ROOT=$v $PY -m pipeline.plan run fetch
  RERUN_ROOT=$v $PY -m pipeline.plan run data
  RERUN_ROOT=$v $rj transform_devign_test -- $PY -m pipeline.transforms.build --dataset devign --split test --workers 4
  RERUN_ROOT=$v $rj graphs_devign_test -- $PY -m pipeline.graphs.build --dataset devign --split test \
      --conditions clean --limit 500 --jobs 2   # = reference chunk 0 (Joern output depends on chunk membership)
  RERUN_ROOT=$v $rj tfidf_devign -- $PY -m pipeline.models.tfidf_lr --dataset devign
  RERUN_ROOT=$v $PY -m pipeline.eval.metrics --dataset devign --model tfidf_lr >/dev/null
  nice -n 19 taskset -c "$RERUN_CPUS" $PY -m pipeline.report.compare --ref "$ref" --new "$v" --graph-slice | tee "$v/compare.txt"
}

# DeepWukong case study: own venv + JDK 11 (env/install_deepwukong.sh); Data.7z must be downloaded by hand.
deepwukong() {
  local D=$RERUN_ROOT/data/deepwukong P=$RERUN_ROOT/venvs/dwk/bin/python rj=pipeline/common/runjob.sh
  [[ -f $D/raw/Data.7z ]] || { echo "put the authors' Data.7z (DeepWukong README, 'Preprocessed CWE119 data') in $D/raw/"; exit 1; }
  env/install_deepwukong.sh
  echo "d0cc5ceabff06ece94ad3bb728241f5b9b2bb7f9cf37949cf542695325b5c869  $D/raw/Data.7z" | sha256sum -c -
  if [[ ! -f $D/upstream_data/CWE119/test.json ]]; then
    mkdir -p "$D/upstream_data" && (cd "$D/upstream_data" && nice -n 19 ionice -c3 taskset -c "$RERUN_CPUS" 7z x -mmt4 -bd -y ../raw/Data.7z >/dev/null)
  fi
  export RERUN_PY=$P
  done_() { grep -q '"exit_code": 0' "$RERUN_ROOT/runs/manifests/$1.json" 2>/dev/null; }   # resumable
  for c in clean identifier deadcode controlflow ren_dead ren_cf dead_cf compound; do
    done_ dwk_prepare_$c || $rj dwk_prepare_$c -- $P -m pipeline.deepwukong.prepare --condition $c --workers 4
    done_ dwk_xfg_$c || $rj dwk_xfg_$c -- $P -m pipeline.deepwukong.xfg --condition $c --jobs 2
  done
  for s in 42 7 100 999 1337; do      # one guarded GPU at a time; waits while none qualifies
    done_ dwk_train_s$s || until RERUN_CPUS=${RERUN_GPU_JOB_CPUS:-22-24} RERUN_THREADS=3 $rj --gpus 1 dwk_train_s$s -- $P -m pipeline.deepwukong.train --seed $s; do
      [[ $? -eq 75 ]] || exit 1; sleep 600; done
    done_ dwk_predict_s$s || $rj dwk_predict_s$s -- $P -m pipeline.deepwukong.predict --seed $s --device cpu
  done
  $rj dwk_report -- $P -m pipeline.deepwukong.report
}

# Every neural model, 1 epoch on a tiny Devign subset, on CPU; nothing is saved.
smoke() {
  local dataset=${RERUN_DATASETS:-devign}
  [[ $dataset != *,* ]] || { echo "smoke: select one RERUN_DATASETS dataset"; return 2; }
  local run="nice -n 19 taskset -c $RERUN_CPUS $PY -m"
  for m in codebert codet5plus; do echo "-- $m"; $run pipeline.models.transformer --model $m --dataset "$dataset" --seed 42 --smoke; done
  echo "-- regvd"; $run pipeline.models.regvd --dataset "$dataset" --seed 42 --smoke
  echo "-- lmggnn"; $run pipeline.models.lmggnn --dataset "$dataset" --seed 42 --smoke
  for m in reveal reveal_struct devign_ggnn ecg_rgcn angle vulgnn vulgnn_struct_cb; do
    echo "-- $m"; $run pipeline.models.graph_train --model $m --dataset "$dataset" --seed 42 --smoke
  done
  echo "smoke: all models OK"
}

stage() {
  case $1 in
    check) check ;;
    test) nice -n 19 $PY -m unittest discover -s tests -t . -v ;;
    fetch|data|transforms|graphs|cpu_models) $PY -m pipeline.plan run "$1" ;;
    gpu_models) $PY -m pipeline.plan enqueue; start_queue ;;
    results) nice -n 19 taskset -c "$RERUN_CPUS" $PY -m pipeline.report.refresh ;;
    status) RERUN_PY=$PY pipeline/status.sh ;;
    verify) verify "${2:-}" ;;
    smoke) smoke ;;
    deepwukong) deepwukong ;;
    all) for s in check fetch data transforms graphs cpu_models gpu_models; do echo "== $s"; stage $s; done ;;
    *) sed -n 2,23p "$0"; exit 2 ;;
  esac
}
stage "${1:-help}" "${2:-}"
