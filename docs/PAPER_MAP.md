# Where each paper result comes from

Every table, figure and in-text number is generated from saved per-sample predictions. None is copied by hand.
`./reproduce.sh results` regenerates all of them (≈6 min on 4 cores); `./reproduce.sh deepwukong` builds the
case study. Paths are relative to `$RERUN_ROOT`.

| Paper element | Generated file | Code | Inputs |
|---|---|---|---|
| Table I (systems, hyperparameters) | `results/tables/table_hparams.tex` | `pipeline/report/tables.py:hparams_table` | `configs/*.yaml` |
| Transformation validity (change / parse / compile rates) | `results/tables/table_validity.{tex,csv}` | `tables.py:validity_table` | `data/<ds>/validity/test/summary.json` |
| Robustness tables (Devign, DiverseVul, Big-Vul) | `results/tables/table_robustness_<ds>.tex` | `tables.py:robustness_table` | `results/stats/<ds>_valid.json` |
| Same, all samples (unchanged or invalid included) | `results/stats/<ds>_all.{json,csv}` | `pipeline/eval/stats.py` | `results/metrics/` |
| Δ precision / recall / PR-AUC / ROC-AUC, calibration (ECE, Brier), 95% CI, raw and BH p | `results/stats/<ds>_valid.csv` | `pipeline/eval/stats.py`, `metrics.py` | `results/preds/` |
| ΔF1 heatmaps | `results/figures/fig_heatmap_<ds>.pdf` | `pipeline/report/analyses.py:heatmaps` | stats |
| Clean F1 vs. Max\|Δ\| correlation | `results/tables/correlation.json` | `analyses.py:correlation` | stats |
| Surface-pattern density | `results/tables/table_pattern_density.{tex,csv}` | `analyses.py:pattern_density` | clean training splits |
| Instance-level prediction flips (mechanism of Δ) | `results/tables/table_flips_<ds>.{tex,csv}` | `analyses.py:flips_table` | preds + validity |
| Control-flow conditions without fallback-only rewrites (sensitivity) | `results/tables/table_cf_sensitivity_<ds>.{tex,csv}` | `analyses.py:cf_sensitivity_table` | preds + validity |
| Identifier-exposure ablation | `results/tables/table_ablation_<ds>.{tex,csv}` | `analyses.py:ablation_table` | metrics + stats |
| Minimum-budget audit (oracle) and fixed-policy attack | `results/tables/table_minbudget_<ds>.{tex,csv}`, `results/attacks/<ds>/minbudget_<model>.json` | `pipeline/eval/attacks.py`, `analyses.py:minbudget_table` | preds + validity |
| Transfer (TASR) | `results/tables/table_transfer_<ds>.{tex,csv}`, `results/attacks/<ds>/transfer.json` | `attacks.py:transfer`, `analyses.py:transfer_table` | preds + validity |
| Robustness training, volume control, held-out family | `results/tables/table_mitigation_devign.{tex,csv}` | `analyses.py:mitigation_table` | stats |
| DeepWukong detection deltas | `results/tables/table_deepwukong.{tex,csv}` | `pipeline/deepwukong/report.py` | `results/preds/deepwukong/` |
| DeepWukong pipeline instrumentation (seed loss, slice change, pollution, topology) | `results/tables/table_deepwukong_pipeline.{tex,csv}` | `pipeline/deepwukong/report.py` | `data/deepwukong/work/<cond>/index.jsonl` |
| Manual semantic audit (90 seeded diffs) | `results/audit/audit_sample.md` | `pipeline/transforms/audit_sample.py` | transformed test sets |
| Executable semantic tests (12 programs × 7 conditions) | stdout of `python -m pipeline.transforms.semantic_tests` | `pipeline/transforms/semantic_tests.py` | none |

## Definitions used by the tables

- **ΔF1** = F1(transformed) − F1(clean) for the same checkpoint and the validation-tuned threshold. It is computed
  on the test functions whose transformation for that condition is *valid*: changed, no new parse errors and no new
  compiler errors. The clean score is recomputed on exactly those ids. The `all` scope keeps every function.
- **Clean F1** in the robustness and mitigation tables is on the full clean test split, mean ± sd over runs.
- **Significance.** Neural detectors: paired t-test over the 5 seed-level deltas, with Benjamini–Hochberg at q = 0.05
  within each table (`bh_family`: main detectors, node-type ablation, training controls). Non-neural detectors
  (30 runs on different 5,000-function training subsamples): the 95% percentile interval of the per-run deltas
  excludes 0.
- **ASR@k** (oracle audit, not an attack under the no-query threat model): among vulnerable test functions the
  detector flags when clean (denominator `|X_f|`), the share evaded by some *valid* condition using ≤ k
  transformation families. Ties are broken by fewest added tokens. The edit cost of the chosen sequence
  (identifiers renamed, occurrences, sites, tokens added) is reported.
- **Fixed-policy ASR** (realisable without queries): the attacker always applies one condition. Its denominator is
  `|X_f|` restricted to functions with a valid transformation.
- **TASR**: on functions both detectors classify correctly, the share of source-evaded functions also evaded at
  the target by the source's minimal sequence. Runs are paired by seed. A neural run is paired with a
  non-neural run positionally. Denominators are in the csv.

## Provenance

Each prediction file records the git commit it was produced at (`meta.git_commit`). Each job has a manifest in
`runs/manifests/` with the command, start/end time, GPU, memory cap, `pip freeze` hash and log path. Only the
REVEAL phase-2 code changed after the other neural runs finished, and all REVEAL-based runs were redone after
that change (see `docs/CHANGES_FROM_V1.md`). Manifests written before 2026-11-07 set `git_dirty` whenever any
file was untracked, including the untracked paper sources, so the flag carries no information for those runs.
Manifests from 2026-11-07 on set it from tracked files only and record a hash of `git diff HEAD`.
