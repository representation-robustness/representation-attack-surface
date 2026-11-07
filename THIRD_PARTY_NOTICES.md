# Third-party code

| Component | Upstream (pinned commit) | License | How it is used |
|---|---|---|---|
| ReVeal phase 2 | github.com/VulDetProject/ReVeal (ca31b78) | MIT, Copyright (c) 2021 VulDetProject | Network and loss ported into `pipeline/models/graph_train.py` (`MetricLearningModel`, `reveal_phase2`) |
| Vul-LMGGNN | github.com/Vul-LMGNN/vul-LMGGNN (7006eb6) | Apache License 2.0 | Architecture ported and modified in `pipeline/models/lmggnn.py` (changes listed in its docstring and `docs/DEVIATIONS.md`) |
| GNN-ReGVD | github.com/daiquocnguyen/GNN-ReGVD (b5e6657) | No license file in the repository | Not redistributed; imported at runtime from a checkout made by `env/install_env.sh` |
| DeepWukong | github.com/jumormt/DeepWukong (28449e5) | MIT, Copyright (c) 2021 DeepWukong | Not redistributed; run unmodified from a checkout |
| Devign | github.com/epicosy/devign (da611bf) | MIT, Copyright (c) 2020 Eduard | Reference only |
| CodeXGLUE | github.com/microsoft/CodeXGLUE (ac74a62) | MIT (code); C-UDA (data) | Devign split and fine-tuning recipe |
| Pretrained models | microsoft/codebert-base, microsoft/graphcodebert-base, Salesforce/codet5p-220m (revisions in `env/upstream.lock`) | see each model card | Downloaded at run time |

The MIT and Apache-2.0 license texts are those of the upstream repositories.
