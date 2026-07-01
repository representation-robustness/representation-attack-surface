# Representation Assumptions as an Attack Surface in Neural Vulnerability Detection

Artifact for the paper:

> **"Representation Assumptions as an Attack Surface in Neural Vulnerability Detection"**  


---

## Overview

This repository contains all code, experimental scripts, and pre-computed result files for the paper. We evaluate ten neural vulnerability detectors across three benchmarks under seven semantics-preserving code transformations, study cross-dataset generalization, and show that identifier-surface assumptions are an exploitable attack vector.

---

## Repository Layout

```
baselines/          one subdirectory per baseline system
utils/              CPG parsers, obfuscation transforms, aggregation script
scripts/            analysis and figure-generation scripts
experiments/        supplementary experiment scripts (ablations, CI, transfer)
devign_full/        all result JSON files (pre-computed; no raw data)
requirements.txt    pinned Python dependency versions
```

---

## Baselines

| Directory | System | Notes |
|-----------|--------|-------|
| `ecg_rgcn/` | ECG RGCN | Implemented from paper (Pativada et al. 2025 — no public code) |
| `angle/` | ANGLE | Implemented from paper (Peng et al. 2024 — no public code) |
| `vulgnn/` | VulGNN | Implemented from paper (Farmer et al. 2026 — no public code) |
| `reveal/` | REVEAL | Faithful reimplementation (Chakraborty et al. 2022 ) |
| `devign_ggnn/` | Devign GGNN | Adapted from authors' GitHub repo |
| `vul_lmggnn/` | Vul-LMGGNN | Wrapper scripts for Liu et al. 2025 (clone to `~/vul-LMGGNN/`) |
| `regvd/` | ReGVD | Implemented from paper (Nguyen et al. 2022 ) |
| `codebert/` | CodeBERT | Fine-tuned via HuggingFace `microsoft/codebert-base` |
| `codet5plus/` | CodeT5+ | Fine-tuned via HuggingFace `Salesforce/codet5p-220m` |
| `tfidf_logreg/` | TF-IDF + LogReg | Implemented from scratch |
| `cpg_logreg/` | CPG + LogReg | Implemented from scratch using Joern CPG features |

> **CodeBERT-Aug** (augmented training variant) lives in `baselines/codebert/codebert_augmented_multiseed.py`.

> **Devign GGNN** collapses to an all-positive predictor on Devign (F1 ≈ 62.7%, Recall ≈ 100%). This is a known replication issue documented in Chakraborty et al. (2022).

---

## Environment Setup

All experiments use Python 3.10. **Prerequisites:** Python 3.10, NVIDIA GPU with CUDA 12.1 drivers.

```bash
# 1. Create the virtual environment
python3.10 -m venv ~/.venvs/vuln-detect
source ~/.venvs/vuln-detect/bin/activate

# 2. Install PyTorch 2.5.1 with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric (prebuilt wheels required)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric==2.7.0

# 4. Install all remaining dependencies
pip install -r requirements.txt
```

If your CUDA version differs, change `cu121` in both URLs above.
See https://pytorch.org/get-started and https://pytorch-geometric.readthedocs.io for alternatives.

All commands below assume the environment is active:
```bash
source ~/.venvs/vuln-detect/bin/activate
```

---

## Dataset Setup



Download and extract the archive (~105 GB uncompressed), then place the contents:

```
dataset_release/
├── in_repo/            ← goes inside the cloned repo directory
│   ├── bigvul/
│   │   └── splits/    (train.jsonl, valid.jsonl, test.jsonl, test_obf_*.jsonl)
│   ├── diversevul_dataset/
│   │   └── splits/    (same structure)
│   └── devign_full/
│       └── devign_input/   (Devign GGNN-format graphs, ~29 GB)
└── home_dir/           ← goes in your home directory ~/
    ├── bigvul_cpg/     (~47 GB, CPG graphs for BigVul GNN baselines)
    ├── diversevul_cpg/ (~25 GB, CPG graphs for DiverseVul GNN baselines)
    └── reveal_cpg/     (~4 GB,  CPG graphs for REVEAL dataset)
```

```bash
# --- in-repo datasets (run from inside the cloned repo directory) ---
cp -r dataset_release/in_repo/bigvul .
cp -r dataset_release/in_repo/diversevul_dataset .
cp -r dataset_release/in_repo/devign_full/devign_input devign_full/

# --- home-directory CPG graphs ---
cp -r dataset_release/home_dir/bigvul_cpg ~/
cp -r dataset_release/home_dir/diversevul_cpg ~/
cp -r dataset_release/home_dir/reveal_cpg ~/
```

The CPG graph paths (`~/bigvul_cpg/`, `~/diversevul_cpg/`, `~/reveal_cpg/`) are hardcoded in
`utils/bigvul_cpg_parser.py`, `utils/diversevul_cpg_parser.py`, and `utils/reveal_cpg_parser.py`.

---

## Generating Obfuscated Test Sets

The seven obfuscated test conditions must exist as `test_obf_*.jsonl` files in each dataset's splits directory. If the Zenodo download already includes them, skip this step.

**Big-Vul and DiverseVul** (JSONL with `func` field):

```bash
python scripts/create_obf_splits.py bigvul/splits/
python scripts/create_obf_splits.py diversevul_dataset/splits/
```

This writes seven variants into the splits directory:
`test_obf_identifier.jsonl`, `test_obf_deadcode.jsonl`, `test_obf_controlflow.jsonl`,
`test_obf_ren_dead.jsonl`, `test_obf_ren_cf.jsonl`, `test_obf_dead_cf.jsonl`, `test_obf_compound.jsonl`.

**Devign** obfuscated inputs are pre-built graph files distributed inside `devign_full/devign_input/obf_*/`.

---

## Obfuscation Conditions

| Key | Condition |
|-----|-----------|
| `original` | Clean — no transformation |
| `identifier` | Identifier renaming (local vars + params → `__v_NNNN`) |
| `deadcode` | Dead-code insertion (opaque predicates, unreachable blocks) |
| `controlflow` | Control-flow restructuring (branch splitting, loop rewriting) |
| `ren_dead` | Identifier renaming + dead-code |
| `ren_cf` | Identifier renaming + control-flow |
| `dead_cf` | Dead-code + control-flow |
| `compound` | All three combined |

---

## Running Each Baseline

Run all commands from the repo root with the virtual environment active.
Results are written to `devign_full/` as `<baseline>_multiseed_results.json`.

### 1. ECG RGCN

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/ecg_rgcn/ecgrgcn_multiseed.py      # Devign (~2 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/ecg_rgcn/ecgrgcn_bigvul.py         # Big-Vul (~6 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/ecg_rgcn/ecgrgcn_diversevul.py     # DiverseVul (~3 hrs)
```

### 2. ANGLE

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/angle/run_multiseed.py             # Devign (~3 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/angle/angle_bigvul.py              # Big-Vul (~8 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/angle/angle_diversevul.py          # DiverseVul (~4 hrs)
```

### 3. VulGNN

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/vulgnn/run_multiseed.py            # Devign (~2 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/vulgnn/vulgnn_bigvul.py            # Big-Vul (~6 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/vulgnn/vulgnn_diversevul.py        # DiverseVul (~3 hrs)
```

### 4. REVEAL

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/reveal/train_reveal_faithful.py    # Devign
CUDA_VISIBLE_DEVICES=0 python baselines/reveal/reveal_bigvul.py            # Big-Vul (~6 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/reveal/reveal_diversevul.py        # DiverseVul (~4 hrs)
```

### 5. Devign GGNN

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/devign_ggnn/devign_ggnn_multiseed.py  # Devign only (~12 hrs)
```

### 6. Vul-LMGGNN

Requires the authors' code cloned to `~/vul-LMGGNN/` (see External Repositories below).

```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/vul_lmggnn/devign_multiseed.sh              # Devign (~20 hrs)
CUDA_VISIBLE_DEVICES=0 bash baselines/vul_lmggnn/run_lmggnn_bigvul_multiseed.sh   # Big-Vul (~24 hrs)
CUDA_VISIBLE_DEVICES=0 bash baselines/vul_lmggnn/run_lmggnn_diversevul_multiseed.sh  # DiverseVul (~12 hrs)
```

### 7. ReGVD

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/regvd/regvd_multiseed.py           # Devign (~6 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/regvd/bigvul_regvd_multiseed.py    # Big-Vul (~8 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/regvd/diversevul_regvd_multiseed.py  # DiverseVul (~4 hrs)
```

### 8. CodeBERT

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/codebert/codebert_multiseed.py              # Devign (~4 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/codebert/codebert_augmented_multiseed.py    # Devign-Aug (~16 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/codebert/bigvul_codebert_multiseed.py       # Big-Vul (~6 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/codebert/diversevul_codebert_multiseed.py   # DiverseVul (~4 hrs)
```

### 9. CodeT5+

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/codet5plus/codet5plus_multiseed.py          # Devign (~5 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/codet5plus/bigvul_codet5plus_multiseed.py   # Big-Vul (~8 hrs)
CUDA_VISIBLE_DEVICES=0 python baselines/codet5plus/diversevul_codet5plus_multiseed.py  # DiverseVul (~5 hrs)
```

### 10. TF-IDF + Logistic Regression

```bash
python baselines/tfidf_logreg/run_tfidf_7cond_devign.py    # Devign (all 7 conditions, 30 trials)
python baselines/tfidf_logreg/bigvul_tfidf_eval.py         # Big-Vul
python baselines/tfidf_logreg/run_tfidf_eval.py            # DiverseVul
```

### 11. CPG + Logistic Regression

Requires Joern installed and `~/bigvul_cpg/` / `~/diversevul_cpg/` populated.

```bash
python baselines/cpg_logreg/run_multi_classifier_eval.py   # Devign
python baselines/cpg_logreg/cpglr_bigvul.py                # Big-Vul
python baselines/cpg_logreg/cpglr_diversevul_7cond.py      # DiverseVul
```

---

## Getting Full 7-Condition Results (GNN Models)

The multiseed training scripts evaluate on 4 conditions only: `test`, `test_obf_identifier`, `test_obf_deadcode`, `test_obf_controlflow`. They also save one checkpoint per seed.

To produce the remaining 4 pairwise/compound conditions, run after training:

```bash
# Big-Vul
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_bigvul_gnns.py --model ecgrgcn
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_bigvul_gnns.py --model angle
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_bigvul_gnns.py --model vulgnn
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_bigvul_gnns.py --model reveal

# DiverseVul
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_diversevul_gnns.py --model ecgrgcn
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_diversevul_gnns.py --model angle
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_diversevul_gnns.py --model vulgnn
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_diversevul_gnns.py --model reveal

# Devign
CUDA_VISIBLE_DEVICES=0 python utils/eval_7cond_devign_gnns.py --model ecgrgcn
```

Each script reads checkpoints from directories created by the multiseed scripts:

| Script | Checkpoint directory |
|--------|---------------------|
| `eval_7cond_bigvul_gnns.py --model ecgrgcn` | `~/ecgrgcn_bigvul_ckpts/` |
| `eval_7cond_bigvul_gnns.py --model angle` | `~/angle_bigvul_ckpts/` |
| `eval_7cond_bigvul_gnns.py --model vulgnn` | `~/vulgnn_bigvul_ckpts/` |
| `eval_7cond_bigvul_gnns.py --model reveal` | `~/reveal_bigvul_ckpts/` |
| `eval_7cond_diversevul_gnns.py --model ecgrgcn` | `~/ecgrgcn_diversevul_ckpts/` |
| `eval_7cond_diversevul_gnns.py --model angle` | `~/angle_diversevul_ckpts/` |
| `eval_7cond_diversevul_gnns.py --model vulgnn` | `~/vulgnn_diversevul_ckpts/` |
| `eval_7cond_diversevul_gnns.py --model reveal` | `~/reveal_sys_diversevul_ckpts/` |
| `eval_7cond_devign_gnns.py --model ecgrgcn` | `~/ecgrgcn_devign_ckpts/` |

Results are written to `devign_full/<model>_7cond_<dataset>_results.json`.

---

## Aggregating Results

Once all JSON result files are in `devign_full/`, run:

```bash
python utils/aggregate_results.py
```

Prints the full Devign, Big-Vul, and DiverseVul F1 tables (clean F1 ± std, all 7 ΔF1 columns) and outputs LaTeX table snippets.

---

## Attack Experiments

The adversarial attack evaluation lives in `devign_full/attack/`:

```bash
# Greedy identifier-renaming attack
python devign_full/attack/greedy_attack.py

# Lexical variant ASR analysis
python devign_full/attack/lexical_variant_asr.py
```

Pre-computed attack results are in `devign_full/attack/greedy_attack_results.json` and
`devign_full/attack/lexical_variant_asr.json`. Per-model predictions used in transfer
analysis are in `devign_full/attack/preds/`.

---

## Identifier Exposure Ablations

Two ablation arms isolate the effect of identifier-derived features on renaming sensitivity
(Section 6.2 of the paper):

- **REVEAL-StructOnly** — REVEAL retrained with identifier tokens stripped from node features
- **VulGNN-CodeBERT** — VulGNN retrained with CodeBERT token embeddings in place of Word2Vec

```bash
# REVEAL-StructOnly: train (5 seeds) then aggregate
CUDA_VISIBLE_DEVICES=0 python experiments/exp10_reveal_noid.py
python experiments/exp10_reveal_noid_aggregate.py

# VulGNN-CodeBERT: train (5 seeds) then aggregate
CUDA_VISIBLE_DEVICES=0 python experiments/exp12_vulgnn_withid.py
python experiments/exp12_vulgnn_withid_aggregate.py
```

Results: `devign_full/reveal_noid_multiseed_results.json`, `devign_full/vulgnn_withid_multiseed_results.json`

---

## Transfer Attack Analysis

Measures how well attack success on one detector transfers to others (Section 6.4).

```bash
python experiments/exp5_transfer_matrix.py    # ASR transfer matrix (Devign)
python experiments/exp17_transfer_v2.py       # extended transfer analysis
```

Results: `devign_full/transfer_asr_matrix.json`, `devign_full/transfer_v2_results.json`

---

## Distribution Shift Quantification

Measures the token-level distribution shift induced by each obfuscation condition (Section 6.5).

```bash
python experiments/exp2_dist_shift.py
```

Results: `devign_full/distribution_shift_results.json`

---

## Robustness Training Experiments

Augmented and curriculum training variants evaluated as mitigations (Section 6.6).

```bash
# ReGVD with augmented (obfuscation-mixed) training data
CUDA_VISIBLE_DEVICES=0 python experiments/exp6_regvd_aug_train.py
python experiments/exp6_regvd_aug_aggregate.py

# REVEAL with augmented training data
python experiments/exp7_reveal_aug_prep.py           # prepare augmented splits
CUDA_VISIBLE_DEVICES=0 python experiments/exp7_reveal_aug_train.py
python experiments/exp7_reveal_aug_aggregate.py

# CodeBERT mixed-training variant
CUDA_VISIBLE_DEVICES=0 python experiments/exp8_codebert_mixed_train.py
python experiments/exp8_codebert_mixed_aggregate.py

# CodeBERT curriculum-training variant
CUDA_VISIBLE_DEVICES=0 python experiments/exp9_codebert_curriculum_train.py
python experiments/exp9_aggregate.py
```

Results: `devign_full/regvd_aug_multiseed_results.json`, `devign_full/reveal_aug_multiseed_results.json`,
`devign_full/codebert_mixed_multiseed_results.json`, `devign_full/codebert_curriculum_multiseed_results.json`

---

## Statistical Tests

Bootstrap confidence intervals and pairwise significance tests for all reported F1 scores.

```bash
python experiments/exp15_bootstrap_ci_v2.py    # 95% CIs via bootstrap resampling
python experiments/exp16_significance_tests.py # pairwise Wilcoxon tests across seeds
```

Results: `devign_full/bootstrap_ci_v2.json`, `devign_full/significance_tests.json`

---

## Generating Figures

```bash
python scripts/generate_heatmaps.py       # per-dataset ΔF1 heatmaps
python scripts/generate_figures.py        # bar charts and scatter plots
python scripts/plot_roc_curves.py         # ROC curves
```

Output goes to `figures/`.

---

## DeepWukong Case Study

DeepWukong must be cloned separately to `~/DeepWukong/`.

```bash
cd ~/DeepWukong
CUDA_VISIBLE_DEVICES=0 python run_clean_train.py        # re-train on SARD CWE-119
CUDA_VISIBLE_DEVICES=0 python generate_roc_and_metrics.py  # inference on all conditions
```

Results: `devign_full/roc_data_dwk.json`

---

## External Repositories

| Repo | Clone to | Used by |
|------|----------|---------|
| Vul-LMGGNN (Liu et al. 2025) | `~/vul-LMGGNN/` | `baselines/vul_lmggnn/` wrapper scripts |
| DeepWukong | `~/DeepWukong/` | case study |

CPG graphs pre-built with Joern are distributed via Zenodo under `home_dir/`.

---

## Pre-computed Results

All result JSON files are checked into `devign_full/` so the tables and figures can be
reproduced without re-running training. To regenerate figures only:

```bash
python scripts/generate_heatmaps.py
python scripts/generate_figures.py
python scripts/plot_roc_curves.py
python scripts/generate_roc_renaming.py
```

---

## License

Code released under the MIT License. See [LICENSE](LICENSE).

Dataset splits derived from:
- **Devign** — Zhou et al. 2019 (original licenses apply)
- **Big-Vul** — Fan et al. 2020 (CC BY 4.0)
- **DiverseVul** — Chen et al. 2023 (original licenses apply)
