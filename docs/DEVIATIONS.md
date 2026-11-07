# Implementation fidelity and deviations

The study compares detectors under one protocol. The following settings are therefore shared by every detector,
even where the original papers differ:

- the same canonical train / validation / test splits (`pipeline/data/prepare.py`);
- one Joern version (v4.0.655) for every graph;
- checkpoint selection by validation F1, with the decision threshold tuned on validation;
- 5 seeds (42, 7, 100, 999, 1337).

Everything else follows the original work as listed below. Upstream code is pinned in `env/upstream.lock`.

| Detector | Code used | Deviations from the original |
|---|---|---|
| ReGVD | Authors' model code imported at runtime from the pinned checkout (`GNN-ReGVD` b5e6657) | Checkpoint by validation F1 instead of accuracy; early-stopping patience 10. |
| Vul-LMGGNN | Ported from `vul-LMGGNN` 7006eb6 (Apache-2.0) into `pipeline/models/lmggnn.py` | Upstream's graph readout applies softmax over a size-1 dimension (`models/layers.py:58` and `:88`), so the graph branch is the constant 1.0 and the released model reduces to CodeBERT. Here the readout emits two logits, and the two branches are mixed as probabilities as the paper describes (k = 0.1). Graphs come from our Joern export instead of the authors' preprocessing. |
| REVEAL | Phase 1: re-implemented gated GNN on statement-level CPG views. Phase 2: ported from `ReVeal` ca31b78 (MIT), `Vuld_SySe/representation_learning/models.py` | Phase 1 is trained with focal loss (γ = 2) instead of cross-entropy. Phase 2 (SMOTE, metric-learning network, loss with λ1 = 0.5, λ2 = 1e-3, margin 0.5, batch 128, patience 5) follows upstream. Phase-2 selection uses our validation split. |
| Devign GGNN | Re-implemented (6-step gated GNN over AST/CFG/data flow, gated readout) | Joern v4 graphs instead of the original Devign graph builder. |
| ECG RGCN, ANGLE, VulGNN | Re-implemented from the papers (`pipeline/models/graph_models.py`) | No released code was used. Node lexical features are the mean Word2Vec of each node's code tokens (100-d, trained on the clean training split), which approximates each paper's token features. |
| CodeBERT | CodeXGLUE defect-detection recipe, `microsoft/codebert-base` pinned | Checkpoint by validation F1. |
| CodeT5+ | `Salesforce/codet5p-220m` encoder, mean pooling, pinned | Checkpoint by validation F1. |
| TF-IDF+LR, CPG+LR | Ours | 30 runs on stratified 5,000-function training subsamples. CPG+LR uses node/edge-type histograms on every dataset. |
| DeepWukong | Unmodified upstream `DeepWukong` 28449e5 (MIT): Joern, slicing, symbolization, training | Only the seed and `num_workers` are set. Transformed XFGs are built by the same upstream code from transformed SARD sources (`pipeline/deepwukong/`). |

## Representation variants (ablation, not separate systems)

`*_struct` variants feed **node types only**. `vulgnn_struct_cb` adds a frozen function-level CodeBERT `[CLS]`
vector before the classifier. These are the controlled arms of the identifier-exposure ablation: the same
architecture and training, with only the lexical channel removed or added.

## Known limitations of the transformations

- Fresh identifiers have the form `__v_NNNN`. They are out of vocabulary for Word2Vec-based models by
  construction: about 21–25% of nodes lose their lexical vector, against 3–5% on clean code.
  Renaming effects for those models therefore include a vocabulary-shift component.
- Dead-code insertion uses `(v ^ v)` only when the function has a plain integer parameter `v`. Otherwise the
  predicate is the literal `0`. Test functions using the literal: Devign 58.6%, DiverseVul 68.1%, Big-Vul 74.9%.
- Control-flow restructuring applies branch splitting, an opaque conjunct in `while` conditions, call wrapping
  and statement flattening. When none of these applies, a *fallback* inserts one always-true guarded no-op block,
  which is closer to dead code than to restructuring. Fallback-only shares of valid control-flow functions:
  Devign 16.2%, DiverseVul 24.6%, Big-Vul 36.7%. `table_cf_sensitivity_<ds>` reports every control-flow
  condition with these functions excluded.
- Validity means: the function changed, has no new tree-sitter parse error, and has no new `gcc -fsyntax-only`
  error in a differential check. Full semantic equivalence cannot be established without each project's build
  context. It is supported by 84 executable equivalence tests and a 90-item manual audit sample.
