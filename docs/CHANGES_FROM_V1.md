# Changes since the submitted version (v1)

The submitted artifact is preserved verbatim at git tag `v1-icse27-submission`. Its transformation code is also
in `legacy/v1_obf_transforms.py`, so its numbers can be reproduced and compared. Every experiment in the paper
was rerun from scratch with the corrected pipeline in this repository. This file lists what changed, why, and
how each change affects the results.

## Defects in v1 and their fixes

| # | v1 defect | Effect on v1 results | v2 fix |
|---|---|---|---|
| 1 | Transformations silently returned the **unmodified function** when the parser found no function definition (25–33% of Big-Vul / DiverseVul, 5–13% of Devign) | Part of every "transformed" test set was clean code, which shrank or hid Δ | Rewrite on tree-sitter C/C++ (`pipeline/transforms/core.py`). Each function is labelled changed / valid, and robustness is computed on validly transformed functions only (`table_validity`) |
| 2 | Opaque predicates were applied to **pointer variables** (type-substring match), declarations were inserted into braceless bodies, and out-of-scope locals were used; 9–26% of dead-code / control-flow outputs did not compile | Inputs were not semantics-preserving | Predicates use only plain integer parameters (else a literal), and every insertion is brace-wrapped. A differential `gcc -fsyntax-only` gate plus 84 executable equivalence tests are in `tests/`. 99.4–99.96% of changed functions are valid |
| 3 | The graph evaluation scripts fed ECG RGCN, VulGNN and REVEAL **node types only**; ANGLE got node code | The reported renaming invariance of these models came from the loader, not the models | Main rows use each paper's lexical node features. The node-types-only versions are a labelled ablation (`*_struct`) |
| 4 | **Different Devign splits** per detector (CodeXGLUE; 80/10/10 with two seeds; Vul-LMGGNN's own) | Rows not comparable | One canonical split per dataset, shared by all detectors (`pipeline/data/prepare.py`) |
| 5 | Some ± sd values came from a **single run**; REVEAL's Devign pairwise evaluations had failed (F1 ≈ 1%) | Variance and significance not supported | 5 seeds for every neural detector, 30 subsample runs for non-neural. Run counts appear in every table |
| 6 | CPG+LR defined differently on Devign (post-GGNN embeddings) than elsewhere (type histograms) | The Devign ΔRen −6.30 for CPG+LR is not comparable | One structural definition everywhere |
| 7 | REVEAL phase 2 (SMOTE + metric learning) was our own re-implementation and differed from upstream | Near-constant scores | Ported from upstream ReVeal (`docs/DEVIATIONS.md`) |
| 8 | Statistics scripts disagreed with the paper (Bonferroni / Wilcoxon in code, BH in text) | — | One procedure (`pipeline/eval/stats.py`): paired t-test with BH per table, raw and adjusted p, CIs and denominators in `results/stats/` |
| 9 | Minimum-budget analysis presented as an attack under a no-query threat model | — | Reported as an offline oracle audit, next to a realisable fixed-policy attack and edit costs (`table_minbudget_<ds>`) |

Upstream detector code was not modified to change results. ReGVD and DeepWukong run from their pinned upstream
checkouts. The only change to upstream behaviour is documented in `docs/DEVIATIONS.md`: Vul-LMGGNN's
constant graph branch.

## Effect on the paper's findings

Derived from the regenerated tables (`docs/PAPER_MAP.md`).

**Still supported**
- Clean F1 does not predict robustness: there is no significant correlation between clean F1 and Max|Δ| on any
  dataset (`correlation.json`).
- Renaming sensitivity follows lexical exposure. In the controlled ablation, all four architectures lose their
  renaming sensitivity when the lexical channel is removed (ΔRen ≈ 0.00), on all three datasets, and the
  difference is significant (`table_ablation_<ds>`).
- Dead code hurts the transformers among the most: CodeBERT on Devign (−3.5), and CodeBERT / CodeT5+ on Big-Vul
  (about −7), together with Vul-LMGGNN and ANGLE.
- Transformation-augmented training reduces CodeBERT's sensitivity. The added clean-data volume control does not,
  so the effect is not data volume (`table_mitigation_devign`).

**Changed**
- With their published lexical inputs, ECG RGCN, ANGLE, VulGNN and REVEAL are renaming-sensitive. Invariance
  holds only for the node-types-only variants.
- Dead-code insertion no longer improves F1 on Devign or DiverseVul. The v1 "dataset-dependent reversal"
  is not reproduced. On Big-Vul the transformer drops are about −7 F1 (v1: −18).
- Minimum-budget ASR and transfer rates are far lower than in v1: in v1, part of the test set was unchanged, and
  some invalid rewrites changed behaviour.
- Adding a CodeBERT channel to node-types-only VulGNN induces a small renaming effect (−0.43), significant only
  against the base model (`table_ablation_devign`), smaller than v1's −2.39.
- ReGVD-Aug does not reduce sensitivity beyond its volume control (Max|Δ| 1.40 vs 1.37).
  REVEAL-Aug reduces renaming sensitivity (ΔRen −0.82 vs −2.64 for the volume control), less than v1 reported.
- Held-out-family training transfers partially. For example, CodeBERT trained without renamed data still has
  ΔRen −0.74, against −1.26 for clean training and −0.27 for full augmentation (`table_mitigation_devign`).

## DeepWukong

The same upstream DeepWukong pipeline and the same five trained models (clean F1 92.2 ± 2.1) were evaluated on SARD
sources transformed by the v1 transforms (`legacy/`, rows `v1_*`) and by the validated v2 transforms.
Paired ΔF1, mean of 5 seeds (`table_deepwukong.csv`):

| Condition | v2 transforms | v1 transforms | Submitted paper (F1 drop from 91.7) |
|---|---|---|---|
| Ren | −0.25 | −0.26 | −52.7 |
| Dead | −3.12 | −1.77 | −23.7 |
| CF | −19.96 | −11.77 | −48.4 |
| R+D | −3.15 | −1.75 | −74.4 |
| R+CF | −20.14 | −11.89 | −81.5 |
| D+CF | −16.02 | −8.80 | −74.8 |
| Cmp | −16.05 | −8.85 | −84.4 |

The v1 transforms do not reproduce the submitted DeepWukong numbers. Under DeepWukong's symbolization, renaming
is nearly invisible with either transform version, and every seed of every slice is kept (`table_deepwukong_pipeline`).
The submitted numbers came from evaluation scripts that are not part of either artifact (`run_clean_train.py`,
`generate_roc_and_metrics.py`, referenced by the v1 README). The revised paper uses only the v2 numbers.
