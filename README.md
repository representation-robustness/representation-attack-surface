# Representation Assumptions as an Attack Surface in Neural Vulnerability Detection

Artifact for the paper. It evaluates 10 vulnerability detectors (whole-function GNNs, graph + language-model
hybrids, transformers and non-neural baselines) on Devign, Big-Vul and DiverseVul under seven
semantics-preserving transformations: identifier renaming, dead-code insertion, control-flow restructuring,
and their pairwise and compound combinations.

Every table, figure and in-text number in the paper is computed by this repository from saved per-sample
predictions. [docs/PAPER_MAP.md](docs/PAPER_MAP.md) maps each one to the file and code that produce it.

## Quick start

```bash
git clone <repo-url> && cd representation-attack-surface
export RERUN_ROOT=/path/with/60GB/rerun            # all data, logs and results go here (default: ../rerun)
export DATASET_RELEASE=/path/to/dataset_release    # released Big-Vul / DiverseVul splits (in_repo/)
SIF=$RERUN_ROOT/images/repattack-v2.sif            # build: env/build_image.sh (rootless, ~30 min), or copy

./reproduce.sh --sif $SIF check     # tool versions, pinned packages, inputs
./reproduce.sh --sif $SIF test      # unit tests + executable semantic-equivalence tests (< 1 min)
./reproduce.sh --sif $SIF all       # CPU stages (~5 h on 4 cores), then queues every GPU run
./reproduce.sh --sif $SIF status    # progress
./reproduce.sh --sif $SIF results   # metrics, statistics and tables from whatever has finished
```

Without `--sif`, the same commands use a host install (see [docs/REPRODUCE.md](docs/REPRODUCE.md) §5).

## Requirements

| | |
|---|---|
| OS | Linux x86_64. [Apptainer](https://apptainer.org) ≥ 1.3 (rootless is fine) or Docker |
| GPU | NVIDIA, ≥ 24 GB, driver ≥ 530. ~260 GPU-hours for the full matrix (≈ 3 days on 4 GPUs) |
| CPU / disk | 4 cores; ~60 GB under `RERUN_ROOT` plus 5 GB for the image |
| Inputs | Devign is downloaded and checksum-verified automatically. Big-Vul / DiverseVul: `dataset_release/in_repo/` |

## Stages

| Stage | What it does | Time |
|---|---|---|
| `test` | unit tests (statistics, attack definitions, transform invariants) and 84 executable equivalence tests | < 1 min |
| `fetch`, `data` | Devign source file; canonical clean splits with stable ids | < 2 min |
| `transforms` | 7 transformation conditions, parse + gcc compile validity gate, 84 executable semantic tests, audit sample | ~15 min |
| `graphs` | Joern v4.0.655 code property graphs for every split and condition | ~4.5 h |
| `cpu_models` | TF-IDF+LR and CPG+LR, 30 subsampled runs each | ~30 min |
| `smoke` | every neural model, 1 epoch on a tiny Devign subset, CPU; checks the environment before GPU time is spent | ~10 min |
| `gpu_models` | queues every neural run (5 seeds) and starts the GPU queue daemon | ~260 GPU-h |
| `results` | metrics → paired t-tests + Benjamini–Hochberg → attacks and transfer → every paper table and figure | ~6 min |
| `verify [REF]` | reruns a deterministic slice and compares it id by id with a reference `RERUN_ROOT` | ~20 min |

Stages are resumable: finished jobs (exit code 0 in `runs/manifests/`) are skipped.
`python -m pipeline.plan list` prints the full experiment matrix.

## GPUs and shared machines

The queue only starts a job on a GPU with no other user's process on it, never touches other processes, and runs
everything at `nice 19`.

| Variable | Default | Meaning |
|---|---|---|
| `RERUN_MAX_GPUS` | `2` | GPUs held at once (one job per GPU) |
| `RERUN_GPUS` | all | allowed GPU ids, e.g. `0,1,2,3` |
| `RERUN_SHARE_IDLE` | `1` | `0` = never share a GPU, even with another user's long-idle process |
| `RERUN_JOBS_PER_GPU` | `1` | pack up to k of our jobs on one GPU, each capped at 1/k of that GPU's memory budget; a packed job that runs out of memory is retried alone |
| `RERUN_SLOT_CPUS`, `RERUN_SLOT_THREADS` | 3-core groups, `3` | cores and threads of each queue slot |
| `RERUN_PRIORITY_FROM` | none | another `RERUN_ROOT`: start the jobs that took longest there first |
| `RERUN_SEEDS`, `RERUN_DATASETS`, `RERUN_JOBS` | all | run a subset, e.g. to split the matrix across machines |

Example: four specific GPUs, no sharing, only two seeds:

```bash
RERUN_MAX_GPUS=4 RERUN_GPUS=0,1,2,3 RERUN_SHARE_IDLE=0 RERUN_SEEDS=42,7 ./reproduce.sh --sif $SIF gpu_models
```

How jobs are packed and scheduled, and how to check efficiency: [docs/PARALLELISM.md](docs/PARALLELISM.md).
Stop the queue with `kill $(cat $RERUN_ROOT/runs/gpu_queue.pid)`. Running jobs finish; pending ones stay queued.

## Outputs (`$RERUN_ROOT`)

```
results/tables/     table_*.tex / *.csv, as included in the paper (list: docs/PAPER_MAP.md)
results/figures/    figures (PDF)
results/stats/      per test: Δ precision / recall / F1 / PR-AUC, t, p, BH-adjusted p, 95% CI, n
results/attacks/    oracle minimum-budget audit, fixed-policy ASR and transfer, with denominators
results/preds/      per-sample probabilities for every run and condition
data/               clean splits, transformed code, validity records, graphs (all with sha256 manifests)
logs/, runs/        full log and provenance manifest (command, commit, pip-freeze hash, GPU, times) per job
```

## Reproducibility

- **Bit-identical across machines:** splits, transforms, validity labels, graphs, and TF-IDF+LR / CPG+LR.
  `verify` checks these.
- **Within run-to-run noise:** the neural models. Seeds are fixed and pretrained checkpoints pinned to Hugging Face
  commits, but GPU kernels are not bit-exact across hardware. Expect per-seed F1 to vary by tenths of a point.
- **Pinned software:** everything is recorded in [`env/`](env/): JDK and Joern checksums, the pip lock file,
  upstream commits.

## Repository layout

```
reproduce.sh        single entry point
pipeline/           data, transforms, graphs, models, evaluation, report
configs/            model hyperparameters (Table I is generated from these)
env/                Apptainer / Docker recipes, lock files, installer
tests/              unit tests (python -m unittest)
legacy/             the submitted version's transforms, verbatim, only to reproduce v1 numbers
tools/              make_review_package.sh (anonymized package of code + results), full_rerun.example.sh
docs/               PAPER_MAP.md (paper element -> file), DEVIATIONS.md (fidelity to each original detector),
                    CHANGES_FROM_V1.md (what changed since the submitted version and why), REPRODUCE.md
```

The original submission artifact is superseded. Its transforms silently fell back to unmodified code and produced
non-compiling rewrites, and its graph models were fed node types only. [docs/CHANGES_FROM_V1.md](docs/CHANGES_FROM_V1.md)
lists every change and its effect on the results.

## DeepWukong case study

`./reproduce.sh deepwukong` (host only: its own venv and JDK 11, installed by `env/install_deepwukong.sh`).
It needs the authors' preprocessed CWE-119 archive: download `Data.7z` from the DeepWukong README
("Preprocessed CWE119 data") into `$RERUN_ROOT/data/deepwukong/raw/`. The archive is checksum-verified.
It writes `results/tables/table_deepwukong{,_pipeline}.{tex,csv}`. The same run also evaluates the submitted
version's transforms (`legacy/`) through the identical pipeline, rows `v1_*`, so the two can be compared directly.

## Not included

- **PrimeVul.**

## License

See [LICENSE](LICENSE). Third-party code and its licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
