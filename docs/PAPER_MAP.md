# Where each paper result comes from

Every table, figure and in-text number is generated from saved per-sample predictions. None is copied by hand.
`./reproduce.sh results` regenerates all of them (≈6 min on 4 cores); `./reproduce.sh deepwukong` builds the
case study. Paths are relative to `$RERUN_ROOT`.

| Paper element | Generated file | Code | Inputs |
|---|---|---|---|
| Table 1 (systems, hyperparameters, classifier input) | `results/tables/table_hparams.tex` | `pipeline/report/tables.py:hparams_table`; classifier input and token limits from `pipeline/models/graph_views.py`, `transformer.py`, `regvd.py`, `lmggnn.py`, `tfidf_lr.py`, `cpg_lr.py`; year = publication year of each cited system | `configs/*.yaml`, prediction `meta` |
| Table 14 validity and coverage (changed / valid %, located %) | `results/tables/table_validity.{tex,csv}` (from raw counts) | `tables.py:validity_table` | `data/<ds>/validity/test/summary.json` |
| Table 14 rejection counts (Parse, Comp.) | `results/tables/table_validity_rejections.csv` | `analyses.py` | `data/<ds>/validity/test/<cond>.jsonl` |
| Table 15 edit cost | `results/tables/table_edit_cost.{tex,csv}` | `analyses.py:edit_cost_table` | validity + graphs |
| Robustness tables (Devign, DiverseVul, Big-Vul, PrimeVul) | `results/tables/table_robustness_<ds>.tex`; paper layout by `tools/paper_tables/make_paper_tables.py`, `make_table_primevul.py` | `tables.py:robustness_table` | `results/stats/<ds>_valid.json` |
| Same, all samples (unchanged or invalid included) | `results/stats/<ds>_all.{json,csv}` | `pipeline/eval/stats.py` | `results/metrics/` |
| Δ precision / recall / PR-AUC / ROC-AUC, calibration (ECE, Brier), 95% CI, raw and BH p | `results/stats/<ds>_valid.csv` | `pipeline/eval/stats.py`, `metrics.py` | `results/preds/` |
| ΔF1 heatmaps (four-dataset figure) | `results/figures/fig_heatmap_<ds>.pdf`; combined figure `results/figures/fig_heatmaps_all4_small.pdf` (`tools/paper_tables/make_fig_heatmaps4.py`) | `pipeline/report/analyses.py:heatmaps` | stats |
| DeepWukong ROC figure | `results/figures/roc_deepwukong_obfuscation.pdf` | `tools/paper_tables/make_fig_deepwukong.py` | `results/preds/deepwukong/` |
| Whitespace probe (Big-Vul formatting, §6.1) | `results/tables/table_whitespace_probe.{tex,csv}` | `pipeline/eval/whitespace_probe.py` | transformer checkpoints, test split |
| Clean F1 vs. Max\|Δ\| correlation | `results/tables/correlation.json` | `analyses.py:correlation` | stats |
| Surface-pattern density | `results/tables/table_pattern_density.{tex,csv}` | `analyses.py:pattern_density` | clean training splits |
| Instance-level prediction flips (mechanism of Δ) | `results/tables/table_flips_<ds>.{tex,csv}` | `analyses.py:flips_table` | preds + validity |
| Control-flow conditions without fallback-only rewrites (sensitivity) | `results/tables/table_cf_sensitivity_<ds>.{tex,csv}` | `analyses.py:cf_sensitivity_table` | preds + validity |
| Identifier-exposure ablation | `results/tables/table_ablation_<ds>.{tex,csv}` | `analyses.py:ablation_table` | metrics + stats |
| Minimum-budget analysis (oracle) and fixed-policy rates (Table 8; ASR@1, ASR@2, mean k for Devign: Table 9) | `results/tables/table_minbudget_<ds>.{tex,csv}` (`fixed_<c>` = operational rate over X_f; `_coverage`, `_conditional` also given), `results/attacks/<ds>/minbudget_<model>.json` | `pipeline/eval/attacks.py`, `analyses.py:minbudget_table` | preds + validity |
| Transfer (TASR) | `results/tables/table_transfer_<ds>.{tex,csv}`, `results/attacks/<ds>/transfer.json` | `attacks.py:transfer`, `analyses.py:transfer_table` | preds + validity |
| Robustness training, volume control, held-out family | `results/tables/table_mitigation_devign.{tex,csv}` | `analyses.py:mitigation_table` | stats |
| DeepWukong detection deltas | `results/tables/table_deepwukong.{tex,csv}` | `pipeline/deepwukong/report.py` | `results/preds/deepwukong/` |
| DeepWukong pipeline instrumentation (seed loss, slice change, pollution, topology) | `results/tables/table_deepwukong_pipeline.{tex,csv}` | `pipeline/deepwukong/report.py` | `data/deepwukong/work/<cond>/index.jsonl` |
| DeepWukong end-to-end detection per function (any-XFG-flagged rule) | `results/stats/deepwukong_function_level.json` | `pipeline/deepwukong/function_level.py` | `results/preds/deepwukong/`, `data/deepwukong/work/clean/functions.jsonl`, `data/deepwukong/test_index.json` |
| Test-sample uncertainty of ΔF1 (paired bootstrap over functions, Appendix C text) | `results/stats/<ds>_bootstrap.csv` | `pipeline/eval/bootstrap.py` | `results/preds/` + validity |
| Precision / recall / PR-AUC changes and 95% seed intervals (Appendix C, Tables 17-18) | `results/stats/<ds>_valid.csv` (`delta_precision`, `delta_recall`, `delta_pr_auc`, `ci_low`, `ci_high`) | `pipeline/eval/stats.py` | metrics |
| In-text numbers not in a table: literal-0 and fallback shares, graph coverage, function lengths, paired-vs-full clean F1 gap, duplicate leakage and dedup F1 change, operational fixed-policy maxima, coverage on X_f, clean-correct shares, node-type-only vs all-positive F1, node-type-only prediction changes under renaming, ablation BH q, transfer yield/std and pairs above the fixed-policy rate, interval median widths, DeepWukong per-condition F1 std | `results/tables/paper_numbers.{json,md}` | `pipeline/report/paper_numbers.py` | preds, validity, attacks, clean splits |
| PrimeVul results | `results/{preds,attacks,stats,metrics}/primevul*`, `results/tables/*_primevul.*` | same pipeline (see `docs/PROVENANCE.md`) | `data/primevul/` |
| Manual semantic audit (90 seeded diffs) | `results/audit/audit_sample.md` | `pipeline/transforms/audit_sample.py` | transformed test sets |
| Executable semantic tests (12 programs × 7 conditions) | stdout of `python -m pipeline.transforms.semantic_tests` | `pipeline/transforms/semantic_tests.py` | none |

## Definitions used by the tables

- **ΔF1** = F1(transformed) − F1(clean) for the same checkpoint and the validation-tuned threshold. It is computed
  on the test functions whose transformation for that condition is *valid*: changed, no new parse errors and no new
  compiler errors. The clean score is recomputed on exactly those ids. The `all` scope keeps every function.
- **Clean F1** in the robustness and mitigation tables is on the full clean test split, mean ± sd over runs.
  The ablation table (Table 7) and its §6.2 text use the paired renaming subset: the node-type-only clean F1 ranges
  (Big-Vul 24.1–26.7, PrimeVul 20.6–21.7) are `clean_f1` in `table_ablation_<ds>.csv`, and the all-positive
  baselines 11.4 and 5.2 are 2p/(1+p) with p = `n_vul`/`n_samples` of the `identifier` rows in
  `results/stats/<ds>_valid.json` (956/15,839 and 614/22,848). `paper_numbers` (`struct_variants`) gives the same
  quantities on the full test split (all-positive F1 10.81 and 4.97), which the paper does not use.
- **Significance.** Neural detectors: paired t-test over the 5 seed-level deltas, with Benjamini–Hochberg at q = 0.05
  within each table (`bh_family`: main detectors, node-type ablation, training controls; the CodeBERT-Aug row of the
  Devign table belongs to the training-controls family). Non-neural detectors
  (30 runs on different 5,000-function training subsamples): the 95% percentile interval of the per-run deltas
  excludes 0.
- **ASR@k** (oracle audit, not an attack under the no-query threat model): among vulnerable test functions the
  detector flags when clean (denominator `|X_f|`), the share evaded by some *valid* condition using ≤ k
  transformation families. Ties are broken by fewest added tokens as recorded by the transforms, where only
  dead-code insertion records tokens (control-flow edits count as 0). ASR@k and mean k do not depend on ties, and
  breaking ties by the true token difference leaves every TASR unchanged; only the `tokens_added` cost of the
  chosen sequence in `table_minbudget_<ds>.csv` / `minbudget_<model>.json` reflects this counter. Table 15's Tok.
  is the token count after minus before for every condition.
- **Fixed-policy ASR** (realisable without queries): the attacker always applies one condition. As in the paper, the
  reported rate (`fixed_<c>` in `table_minbudget_<ds>.csv`) is over all of `|X_f|`: an invalid or inapplicable
  rewrite leaves the flagged function in place and counts as not evaded. `fixed_<c>_coverage` is the share of `|X_f|`
  with a valid rewrite; `fixed_<c>_conditional`, and the `asr` / `fixed_policy_mean` fields of
  `minbudget_<model>.json`, are the rate conditional on a valid rewrite.
- **TASR**: on functions both detectors classify correctly, the share of source-evaded functions also evaded at
  the target by the source's minimal sequence. Runs are paired by seed. A neural run is paired with a
  non-neural run positionally. Denominators are in the csv.

## Provenance

Each prediction file records the git commit it was produced at (`meta.git_commit`). Each job has a manifest in
`runs/manifests/` with the command, start/end time, GPU, memory cap, `pip freeze` hash and log path. Only the
REVEAL phase-2 code changed after the other neural runs finished, and all REVEAL-based runs were redone after
that change. Manifests written before 2026-11-07 set `git_dirty` whenever any
file was untracked, including the untracked paper sources, so the flag carries no information for those runs.
Manifests from 2026-11-07 on set it from tracked files only and record a hash of `git diff HEAD`.
