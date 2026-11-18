# Provenance: which code produced which result

Every prediction file records the commit it was produced at (`meta.git_commit`), and every job manifest
(`runs/manifests/<job>.json`) records the commit, whether the working tree had uncommitted changes
(`git_dirty`) and, from 2026-11-07, the sha256 of `git diff HEAD` (`git_diff_sha256`).

The runs were made in the development repository. This repository is published as a single anonymized release
commit, so the recorded hashes are development hashes that do not appear in its history. This file states, for
each recorded commit, how its `pipeline/` and `configs/` differ from the release (checked with
`git diff <commit> -- pipeline configs` in the development repository), and how the released results were checked.

## Recorded commits

| Recorded commit | Prediction files | Manifests | Jobs |
|---|---:|---:|---|
| `fab7c63` | 187 | 28 | CodeBERT, CodeT5+, ReGVD, Vul-LMGGNN, TF-IDF+LR, CPG+LR, graphs, DeepWukong |
| `91bd2e5` | 116 | 116 | ANGLE, Devign GGNN, ECG RGCN, REVEAL, VulGNN |
| `eeef60b` | 89 | 89 | ANGLE, CodeBERT, CodeT5+, ECG RGCN, Vul-LMGGNN, ReGVD, VulGNN |
| `0f61bee` | 31 | 30 | ANGLE, CodeBERT, ECG RGCN, REVEAL, VulGNN |
| `ba7a843` | 11 | 7 | CodeBERT, CodeT5+, DeepWukong |
| `98af43c`, `132826e`, `22d55c4` | 10 | 15 | CodeBERT, ReGVD, graphs, transforms, semantic tests, audit sample |
| `ed0dde5` | 1 | 1 | CodeBERT |
| `e8e61ca`, `b5ccfc3`, `7d770f3`, `6ec4b2e` | 0 | 7 | fetch, clean splits, transforms, graphs |
| `7327cbf` (PrimeVul workspace) | 55 | — | PrimeVul: graph models, ReGVD, Vul-LMGGNN, CodeT5+ |
| `292bfca` (PrimeVul workspace) | 65 | — | PrimeVul: CodeBERT, TF-IDF+LR, CPG+LR |

The PrimeVul runs were made in a second workspace with the same pipeline; their per-job manifests and logs are
not included. Three manifests and five prediction files (CodeT5+, ReGVD, graphs) record no commit.

## How each recorded commit differs from the release

Every difference under `pipeline/` and `configs/`, for every recorded commit, falls into these groups:

1. **PrimeVul support** (in the commits that predate it): `configs/models.yaml` adds `primevul` to
   `balanced_sampler_datasets`; `pipeline/data/prepare.py` adds the PrimeVul loader; `pipeline/models/lmggnn.py`
   and `regvd.py` treat `primevul` like `bigvul` (one line each); `pipeline/transforms/audit_sample.py` accepts
   `--datasets` and includes PrimeVul. Nothing changes for Devign, DiverseVul or Big-Vul.
2. **Cache locking** (`pipeline/models/graph_train.py`, `graph_views.py`; in the commits before it was added, and
   `7327cbf`): concurrent jobs share one feature cache through a file lock and an atomic rename. What is computed
   is unchanged.
3. **Job scheduling** (`pipeline/common/gpu_guard.py`, `gpu_queue.py`, `runjob.sh`, `pipeline/plan.py`): GPU
   admission, packing, job order, the `fetch` stage.
4. **Reporting added later** (`pipeline/eval/bootstrap.py`, `whitespace_probe.py`,
   `pipeline/deepwukong/function_level.py`, `pipeline/report/{paper_numbers,refresh,tables,analyses,gpu_usage,
   compare_findings}.py`, docstring of `pipeline/eval/attacks.py`): computed from saved predictions, after training.
5. **Comments and a removed option**: a docstring in `pipeline/transforms/core.py`, and the unused
   `--transforms v1` comparison option of `pipeline/deepwukong/prepare.py` (no reported result used it).

### Runs with uncommitted changes

- `ba7a843+dirty` (18 manifests: CodeT5+, Vul-LMGGNN, ReGVD, DeepWukong training, prediction and report): the
  recorded diff hash `3fafd463…` equals the sha256 of `git diff ba7a843 eeef60b`, so the tree was exactly `eeef60b`.
- `0f61bee+dirty` (1 manifest, `bigvul/reveal_struct/run_1337`): diff hash `14dc914d…` matches no committed state;
  the only file changed between `0f61bee` and the next commit is `pipeline/common/gpu_guard.py` (GPU admission).
- `ed0dde5+dirty` (`dwk_prepare_dead_cf`): diff hash `7164e5c0…`, no committed match; the next commit changes only
  `pipeline/models/graph_train.py` and `graph_views.py` (cache locking), and `pipeline/deepwukong/` is unchanged
  from `ed0dde5` onwards apart from groups 4 and 5.

Manifests written before 2026-11-07 set `git_dirty` whenever any file was untracked, including the untracked paper
sources, so the flag carries no information for those runs.

## Regenerating the reported results

From the predictions, the release regenerates the reported results byte-identically:

- `./reproduce.sh results` writes every stats, metrics, attack and table file of the four datasets: 211 of 211
  files identical to `results/`.
- From the release asset (`./reproduce.sh predictions`, which has no DeepWukong predictions), the same stage gives
  identical files except the `deepwukong_paired_f1_sd` entry of `paper_numbers.{json,md}`, which stays empty.
- `python -m pipeline.deepwukong.report` regenerates the DeepWukong tables and `results/stats/deepwukong.json`.

Not covered by this check: the whitespace probe (re-scores checkpoints on a GPU), and the audit sample, which the
release draws for four datasets while the shipped `results/audit/audit_sample.md` covers Devign, DiverseVul and
Big-Vul.
