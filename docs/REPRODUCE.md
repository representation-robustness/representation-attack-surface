# Reproducing the v2 results

Everything in the paper's tables is produced by one script, `./reproduce.sh`, from pinned software and
checksummed inputs. Numbers are never copied by hand: every table cell is computed from saved per-sample
prediction files (`results/preds/…`) by `pipeline/report/`.

[`PAPER_MAP.md`](PAPER_MAP.md) maps every paper element to its generated file. What changed since the
submitted version and why is in [`CHANGES_FROM_V1.md`](CHANGES_FROM_V1.md); fidelity to each original detector
is in [`DEVIATIONS.md`](DEVIATIONS.md). Run `./reproduce.sh test` first (< 1 min).

## 1. What you need

| | |
|---|---|
| OS | Linux x86_64. Reference machine: Ubuntu 22.04, 7× RTX A6000 (48 GB), 32 cores |
| Container | [Apptainer](https://apptainer.org) ≥ 1.3 (no root needed, see below), **or** Docker, **or** neither (host venv, §5) |
| GPU | any NVIDIA GPU with ≥ 24 GB, driver ≥ 530 (CUDA 12.1 wheels). 2 GPUs by default (`RERUN_MAX_GPUS`) |
| Disk | ~50 GB under `RERUN_ROOT` (data and graphs ~6 GB, feature caches, Hugging Face models, checkpoints) + ~10 GB for the image |
| Inputs | `dataset_release/in_repo/` (released Big-Vul and DiverseVul splits; ask the authors, 19 GB zip). Devign is downloaded automatically |
| Network | GitHub, PyPI, download.pytorch.org, data.pyg.org, Hugging Face, Google Drive (first run only) |

## 2. Get the image

**Option A: build it yourself** (rootless; about 30 minutes):

```bash
git clone <this repo> && cd representation-attack-surface && git checkout rerun-v2
export RERUN_ROOT=/path/with/space/rerun          # all data, logs and results go here
env/build_image.sh                                # -> $RERUN_ROOT/images/repattack-v2.sif
```

`env/build_image.sh` runs `apptainer build --fakeroot --ignore-subuid` on `env/apptainer.def`. It is niced, uses idle I/O and at most 4 cores.
`env/apptainer.def` and `env/Dockerfile` both run the same `env/install_env.sh`. That recipe installs:
- Temurin JDK 21.0.12+8 and Joern v4.0.655, both checksum-verified;
- the exact pip freeze of the reference venv (`env/requirements-main.lock.txt`, CUDA 12.1 wheels);
- the upstream repos at the commits in `env/upstream.lock`.

**Option B: use the prebuilt image.** Copy `repattack-v2.sif` from the authors (same lab server: `$RERUN_ROOT/images/`).
Check it against `repattack-v2.sif.sha256`.

**Apptainer without root.** If `apptainer` isn't installed, extract the official release package into your home directory:

```bash
curl -LO https://github.com/apptainer/apptainer/releases/download/v1.3.6/apptainer_1.3.6_amd64.deb
# sha256 2723b2928cfc30edf687723c49556ec4e013f0bf7cdb43a5a76bca7bd3c70792
dpkg-deb -x apptainer_1.3.6_amd64.deb $RERUN_ROOT/tools/apptainer
cd $RERUN_ROOT/tools/apptainer && mv usr/* . && rmdir usr
sed -i 's/^mksquashfs procs = 0/mksquashfs procs = 4/' etc/apptainer/apptainer.conf   # don't grab every core
```

This needs unprivileged user namespaces (`unshare -Ur true` must succeed), which stock Ubuntu 22.04 has.

**Docker** (if you're in the `docker` group):

```bash
docker build -t repattack:v2 -f env/Dockerfile .
docker run --gpus '"device=0,1"' -v $PWD:/work -v $RERUN_ROOT:$RERUN_ROOT -e RERUN_ROOT=$RERUN_ROOT \
    -v $DATASET_RELEASE:$DATASET_RELEASE:ro -e DATASET_RELEASE=$DATASET_RELEASE repattack:v2 ./reproduce.sh all
```

## 3. Run

```bash
export RERUN_ROOT=/path/with/space/rerun
export DATASET_RELEASE=/path/to/dataset_release
SIF=$RERUN_ROOT/images/repattack-v2.sif

./reproduce.sh --sif $SIF check        # versions, pip freeze vs lock, inputs found
./reproduce.sh --sif $SIF all          # every CPU stage, then queues all GPU runs
./reproduce.sh --sif $SIF status       # any time: progress overview
./reproduce.sh --sif $SIF results      # any time: recompute metrics/stats/tables from finished runs
```

`all` runs these stages in order; you can also run each one alone. Every stage is resumable, and finished jobs are skipped.

| Stage | What it does | Wall-clock in the reference run (4 cores) |
|---|---|---|
| `fetch` | Devign `function.json` from Google Drive, sha256-checked | < 1 min |
| `data` | canonical clean splits with stable ids (Devign = CodeXGLUE split) | < 1 min |
| `transforms` | v3 transforms for 7 conditions + gcc/g++ validity gate + 84 executable semantic tests + 90-diff audit sheet | ~15 min |
| `graphs` | Joern CPGs for every split/condition (largest method per file) | ~4.5 h |
| `cpu_models` | TF-IDF+LR and CPG+LR, 30 subsampled runs each | ~30 min |
| `smoke` | every neural model, 1 epoch on a tiny Devign subset, CPU | ~10 min |
| `gpu_models` | queues 265 neural runs (5 seeds) and starts the GPU queue daemon | ~260 GPU-hours |
| `results` | metrics → statistics → attacks/transfer → tables | 5 min |

`python -m pipeline.plan list` prints every job with its exact command and whether it is done.
The plan module (`pipeline/plan.py`) is the single definition of the experiment matrix.

## 4. Shared-machine behaviour (built in)

The pipeline was written for a shared lab server and is polite by default:
- **GPUs:**
  - A GPU counts as free only when it has *no other user's process*, under 1 GiB of memory used and under 5% utilization over a sampling window (`pipeline/common/gpu_guard.py`).
  - At most `RERUN_MAX_GPUS` GPUs are used (default 2); `RERUN_GPUS=0,1,2,3` restricts which ones.
  - When no GPU is free, a GPU whose other-user processes have been idle may be shared with a hard memory cap; `RERUN_SHARE_IDLE=0` disables this.
  - `RERUN_SEEDS`, `RERUN_DATASETS` and `RERUN_JOBS` (regex) queue only part of the matrix, e.g. to split it across machines.
  - When none is free, jobs wait and are retried every 10 minutes (`pipeline/common/gpu_queue.py`). Nobody else's job is ever touched.
- **CPU:**
  - Every job runs at `nice 19` with idle I/O priority.
  - CPU stages are pinned to 4 cores (`RERUN_CPUS`, default the last 4), and each GPU job to 3 cores (`RERUN_SLOT_CPUS`).
  - Joern's JVM is capped at 2 processors.
- **Stopping:** `kill $(cat $RERUN_ROOT/runs/gpu_queue.pid)` stops the queue. Running jobs finish; pending ones stay queued.

## 5. Without a container (host venv)

The same layout the reference run used:

```bash
python3.10 -m venv $RERUN_ROOT/venvs/main
$RERUN_ROOT/venvs/main/bin/pip install pip==24.2 setuptools==75.1.0 wheel==0.44.0
$RERUN_ROOT/venvs/main/bin/pip install -r env/requirements-main.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# JDK and Joern under $RERUN_ROOT/tools/{jdk,joern-v4.0.655}, upstream repos under $RERUN_ROOT/upstream:
env/install_env.sh /tmp/x && mv /tmp/x/jdk $RERUN_ROOT/tools/jdk && mv /tmp/x/joern-v4.0.655 $RERUN_ROOT/tools/ \
    && mv /tmp/x/upstream $RERUN_ROOT/upstream
./reproduce.sh all
```

## 6. Where things end up (`$RERUN_ROOT`)

```
data/<ds>/clean/                 canonical splits + manifest.json (input/output sha256)
data/<ds>/transformed/<split>/   transformed functions per condition
data/<ds>/validity/<split>/      per-sample validity records + summary.json
data/<ds>/graphs/<split>/        Joern graphs (*.jsonl.gz) + missing.json
results/preds/<ds>/<model>/      run_<k>.json.gz: per-sample probabilities for every test condition
results/metrics, stats, attacks  derived numbers (JSON/CSV)
results/tables/                  table_*.tex / *.csv, exactly what the paper includes
results/audit/audit_sample.md    manual semantic-audit sheet
logs/<job>.log                   full log of every job
runs/manifests/<job>.json        command, exit code, git commit, pip-freeze hash, container, GPUs, times
```

## 7. How close will my numbers be?

- **Bit-identical:**
  - data splits, transforms and validity labels (deterministic, no RNG);
  - graphs (pinned Joern);
  - TF-IDF+LR and CPG+LR (seeded subsamples, CPU).
- **Very close but not bit-identical:**
  - The neural models. Seeds are fixed, but cuDNN/cuBLAS kernels are non-deterministic across GPU models and driver versions.
  - Expect per-seed F1 to differ by tenths of a point. Five-seed means and the significance decisions should agree.
  - Pretrained checkpoints are pinned to exact Hugging Face commits (`pipeline/models/common.py`, `HF_REVISIONS`).

## 8. Not covered by `reproduce.sh` yet

- **DeepWukong** runs on the host only, via `./reproduce.sh deepwukong` (see README). The authors' `Data.7z` has to be downloaded by hand. The stage takes ~3.5 h on 4 cores and ~4 GPU-hours.
- **PrimeVul** uses the original default release; see the README extension instructions.
  Use an isolated `RERUN_ROOT` and `RERUN_DATASETS=primevul`. Pilot artifacts must use a separate root.
