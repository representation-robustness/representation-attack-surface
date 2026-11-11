# Provenance: which code produced which result

Every prediction file records the commit it was produced at (`meta.git_commit`), and every job manifest
(`runs/manifests/<job>.json`) records the commit, whether the working tree had uncommitted changes
(`git_dirty`) and, from 2026-11-07, the sha256 of `git diff HEAD` (`git_diff_sha256`).

The runs were made in the development repository. The public history was published from it with an
anonymous author, so the recorded hashes are development hashes. This table maps each one to the public
commit whose `pipeline/` and `configs/` trees are byte-identical (checked with `git rev-parse <commit>:pipeline`).

## Devign, DiverseVul, Big-Vul, DeepWukong

| Recorded commit | Prediction files | Public commit with identical code | Notes |
|---|---:|---|---|
| `fab7c63` | 187 | `292bfca` | |
| `91bd2e5` | 116 | `1faad3f` | |
| `eeef60b` | 89 | `eafff15` | |
| `0f61bee` | 30 | `67c98bc` | |
| `ba7a843` | 2 | `d002968` | |
| `ba7a843+dirty` | 9 (+ DeepWukong training, prediction and report) | `eafff15` | The recorded diff hash `3fafd463…` equals the sha256 of `git diff ba7a843 eeef60b`: the tree was exactly `eeef60b` |
| `ed0dde5` | 1 | `a5f6c4a` | |
| `98af43c` | 6 | none; equal to `292bfca` except job scheduling | Differs only in `pipeline/common/gpu_guard.py`, `gpu_queue.py`, `plan.py` (job order) and `pipeline/report/gpu_usage.py` |
| `132826e` | 3 | none; equal to `292bfca` except job scheduling | As above |
| `22d55c4` | 1 | none; equal to `292bfca` except job scheduling | As above |
| `0f61bee+dirty` | 1 (`bigvul/reveal_struct/run_1337`) | `67c98bc` + uncommitted edits | Diff hash `14dc914d…` matches no committed state. The only file changed between `0f61bee` and the next commit is `pipeline/common/gpu_guard.py` (GPU admission) |

DeepWukong prediction files do not record a commit; their manifests (`dwk_*`) do and are covered above.
One DeepWukong preprocessing job, `dwk_prepare_dead_cf`, ran at `ed0dde5` with uncommitted edits (diff hash
`7164e5c0…`, no committed match). The files changed in the next commit are `pipeline/models/graph_train.py`
and `graph_views.py` (cache locking); `pipeline/deepwukong/` is identical from `ed0dde5` to `1faad3f`.

"Identical except job scheduling" was checked with `git diff <commit> 292bfca -- pipeline/models
pipeline/transforms pipeline/eval pipeline/data pipeline/deepwukong configs pipeline/report/tables.py
pipeline/report/analyses.py`, which is empty for all three.

## PrimeVul

| Recorded commit | Produced in | Public commit | Notes |
|---|---|---|---|
| `7327cbf` | PrimeVul workspace | `2cf9b6a` | Same model, transform, evaluation and report code; the public commit omits a host-specific GPU-admission exception and two host-specific helper files, and adds the newer scheduling and cache-locking commits |

## Regenerating the reported results

Running `./reproduce.sh results` at `2cf9b6a` (the public commit that adds PrimeVul support) or later regenerates the Devign,
DiverseVul and Big-Vul stats, tables and attack results byte-identically from the committed predictions
(160 of 160 files).
