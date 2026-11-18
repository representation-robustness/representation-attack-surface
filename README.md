# Representation Assumptions as an Attack Surface in Neural Vulnerability Detection

Artifact for the paper. It evaluates 10 vulnerability detectors (whole-function GNNs, graph + language-model
hybrids, transformers and non-neural baselines) and their node-type-only variants on Devign, DiverseVul, Big-Vul and
PrimeVul under seven semantics-preserving transformations: identifier renaming, dead-code insertion, control-flow
restructuring, and their pairwise and compound combinations, plus a DeepWukong case study.

Every table, figure and in-text number in the paper is computed by this repository from saved per-sample
predictions. [docs/PAPER_MAP.md](docs/PAPER_MAP.md) maps each one to the file and code that produce it.

## Check the reported numbers (no setup)

Every reported result is in this repository under `results/` as generated tables, statistics and JSON
(the paper's LaTeX layout of the tables and the combined heatmap figure are made from them by `tools/paper_tables/`).
[docs/PAPER_MAP.md](docs/PAPER_MAP.md) maps each table, figure and in-text number to its file, e.g.
`results/tables/table_robustness_devign.tex` (Table 2) or `results/tables/paper_numbers.md` (in-text numbers).

## Quick start (full reproduction)

All inputs come with the repository or are downloaded and checksum-verified: Big-Vul, DiverseVul and PrimeVul
are in `datasets/` (xz, 87 MB), Devign and the upstream detector code are fetched at pinned versions.

```bash
git clone <repo-url> && cd representation-attack-surface   # or unpack the anonymized archive
export RERUN_ROOT=/path/with/space/rerun          # all data, logs and results go here (default: ../rerun)
env/build_image.sh                                 # rootless Apptainer image, ~30 min -> $RERUN_ROOT/images/
SIF=$RERUN_ROOT/images/repattack-v2.sif

./reproduce.sh --sif $SIF check     # tool versions, pinned packages, inputs
./reproduce.sh --sif $SIF test      # unit tests + executable semantic-equivalence tests (< 1 min)
./reproduce.sh --sif $SIF all       # fetch + unpack inputs, CPU stages, then the GPU queue (see times below)
./reproduce.sh --sif $SIF status    # progress
./reproduce.sh --sif $SIF results   # metrics, statistics and tables from whatever has finished
```

Without `--sif`, the same commands use a host install (see [docs/REPRODUCE.md](docs/REPRODUCE.md) §5).

## Rebuild every table without GPUs

The `predictions` stage downloads the per-sample predictions behind the reported results (342 MB release asset,
checksum-verified) instead of training. From them, `results` regenerates byte-identically every metrics, stats and
attack file of the four datasets (including the statistics behind the interval and precision/recall tables, Tables
17–18) and the robustness, ablation, attack, transfer, mitigation, flip, validity and edit-cost tables. It needs
network access (Devign download, the release asset), the image from Quick start, and ~10 GB of disk; the CPU stages take ~5–6 h on 4 cores for Devign, DiverseVul and
Big-Vul (measured), plus several hours for PrimeVul's graphs (not measured on the reference machine):

```bash
for s in check fetch data transforms graphs predictions results; do ./reproduce.sh --sif $SIF $s; done
diff -r results/tables $RERUN_ROOT/results/tables     # run from the repository root: compare with the snapshot
```

Expected differences: `paper_numbers.{json,md}` leave `deepwukong_paired_f1_sd` empty, the DeepWukong and
whitespace-probe tables are only in `results/tables`, and `results/stats/deepwukong*.json` are not produced. Both come from stages this route does not run: DeepWukong
(own data and stage, below) and the whitespace probe (`pipeline/eval/whitespace_probe.py`, which re-scores trained
CodeBERT/CodeT5+ checkpoints on a GPU).

## Requirements

| | |
|---|---|
| OS | Linux x86_64. [Apptainer](https://apptainer.org) ≥ 1.3 (rootless is fine) or Docker |
| GPU | NVIDIA, ≥ 24 GB, driver ≥ 530. Devign, DiverseVul, Big-Vul: 265 runs, ~285 GPU-hours at one GPU per job (reference run). PrimeVul adds 60 runs (not measured on the reference machine) |
| CPU / disk | 4 cores. Full run: ~150 GB under `RERUN_ROOT` for three datasets (feature caches ~105 GB, checkpoints ~25 GB), more with PrimeVul; no-GPU route ~10 GB. Image: 4.7 GB |
| Inputs | Big-Vul, DiverseVul, PrimeVul: shipped in `datasets/`. Devign: downloaded. All checksum-verified ([docs/DATASETS.md](docs/DATASETS.md)) |

## Stages

| Stage | What it does | Time |
|---|---|---|
| `test` | unit tests (statistics, attack definitions, transform invariants) and 84 executable equivalence tests | < 1 min |
| `fetch`, `data` | Devign source file, unpack `datasets/`; canonical clean splits with stable ids | < 5 min |
| `transforms` | 7 transformation conditions, parse + gcc compile validity gate, 84 executable semantic tests, audit sample | ~15 min |
| `graphs` | Joern v4.0.655 code property graphs for every split and condition | ~5 h for three datasets; PrimeVul adds several hours |
| `cpu_models` | TF-IDF+LR and CPG+LR, 30 subsampled runs each | ~30 min |
| `smoke` | every neural model, 1 epoch on a tiny Devign subset, CPU; checks the environment before GPU time is spent | ~10 min |
| `predictions` | instead of `gpu_models`: download the reported predictions (342 MB, checksum-verified) | minutes |
| `gpu_models` | queues every neural run (5 seeds) and starts the GPU queue (foreground inside a container) | see Requirements |
| `results` | metrics → paired t-tests + Benjamini–Hochberg → attacks and transfer → every paper table and figure | ~6 min |
| `verify [REF]` | reruns a deterministic slice and compares it id by id with `REF`, the `RERUN_ROOT` of an earlier complete run of yours (e.g. on another machine) | ~20 min |

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

The reported results are also in this repository: `results/` (tables, stats, metrics, attacks, figures, audit)
and `runs/manifests/`, with paths and host names replaced by placeholders. The per-sample predictions behind them
are a release asset (`./reproduce.sh predictions`).

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
configs/            model hyperparameters (Table 1 is generated from these)
env/                Apptainer / Docker recipes, lock files, installer
tests/              unit tests (python -m unittest)
datasets/           Big-Vul, DiverseVul and PrimeVul inputs (xz) with SHA256SUMS
results/, runs/     the reported results and per-job manifests
tools/              make_review_package.sh (anonymized package of code + results), full_rerun.example.sh
docs/               PAPER_MAP.md (paper element -> file), DEVIATIONS.md (fidelity to each original detector),
                    REPRODUCE.md, PROVENANCE.md, DATASETS.md
```

## DeepWukong case study

`./reproduce.sh deepwukong` (host only: its own venv and JDK 11, installed by `env/install_deepwukong.sh`).
It needs the authors' preprocessed CWE-119 archive: download `Data.7z` from the DeepWukong README
("Preprocessed CWE119 data") into `$RERUN_ROOT/data/deepwukong/raw/`. The archive is checksum-verified.
It writes `results/tables/table_deepwukong{,_pipeline}.{tex,csv}`.

## PrimeVul

PrimeVul uses the authors' original default (unpaired) train/valid/test release, shipped in `datasets/primevul/`
and unpacked to `$RERUN_ROOT/data/primevul/raw/` by the `fetch` stage. It runs with the other datasets; to run it
alone, set `RERUN_DATASETS=primevul`. PrimeVul uses the same class-balanced training sampler as Big-Vul, and the
full original splits are kept (no training subsampling).


## License

See [LICENSE](LICENSE). Third-party code and its licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
