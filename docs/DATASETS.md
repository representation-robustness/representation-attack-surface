# Datasets

Every input is pinned by sha256; `pipeline/data/prepare.py` records input and output checksums in
`$RERUN_ROOT/data/<dataset>/clean/manifest.json`. Counts are functions, with the vulnerable share in brackets.

| Dataset | Train / valid / test | Inputs (sha256) |
|---|---|---|
| Devign | 21,854 (45.8%) / 2,732 (43.4%) / 2,732 (45.9%) | `function.json` `0a3b2d561dc6280e53795886ede727d0045c016d083905ba3e9ce384a7eab246`<br>`train.txt` `f0a25410594302a9f0e542a393ad82ad479308a8aa471f4d6cf61b91d6d572bf`<br>`valid.txt` `9f2fa1e108955f197d4a7fa2aa2c7f5e542457b51e0eb1f6e890172d6f700a6e`<br>`test.txt` `b5336b337170ea1edf0570b69edb5a90e3c99bf41cd92909795f5fe32d376d52` |
| DiverseVul | 29,278 (49.8%) / 3,659 (49.7%) / 3,661 (51.8%) | `train.jsonl` `d4484b17622c223776926056423168a4f9d6fe05ffd8702e28567eefd836515a`<br>`valid.jsonl` `4fefdd62cc9092141648139b5d0c8665f6c8a435812b7ebbea364a56db773b18`<br>`test.jsonl` `e5a0db07cdc227f14470a7729f386f6a7f5c2112fb9d97ba9789d5161cb834e9` |
| Big-Vul | 150,888 (5.8%) / 18,861 (5.8%) / 18,861 (5.8%) | `train.jsonl` `7b89ef2c88619c44ab1c5c06968688a2ebbb45df4921bc5281f458ef8f079cc6`<br>`valid.jsonl` `321bf61ab668973bf040541c485aca4e6cc2c193551aed908ea545488299da79`<br>`test.jsonl` `8259abd1f3b8e2d330c514dd9c862a3d060083ca5e04b3d781a6c6a750ae40b1` |
| PrimeVul | 184,427 (3.0%) / 25,430 (2.7%) / 25,911 (2.7%) | `primevul_{train,valid,test}.jsonl` — TODO: sha256 from the PrimeVul run's `manifest.json` |
| SARD CWE-119 (DeepWukong) | authors' split, 8,922 test XFGs (15.8% vulnerable) | `Data.7z` `d0cc5ceabff06ece94ad3bb728241f5b9b2bb7f9cf37949cf542695325b5c869` |

## Sources

| Dataset | Where it comes from | How the artifact obtains it | Licence |
|---|---|---|---|
| Devign | Zhou et al. 2019 `function.json`, CodeXGLUE defect-detection split indices | Downloaded and checksum-verified by `reproduce.sh` | TODO |
| Big-Vul | Fan et al. 2020; our released train/valid/test split files | `$DATASET_RELEASE/in_repo/bigvul/splits/` (archive) | TODO: check redistribution |
| DiverseVul | Chen et al. 2023; our released balanced split files | `$DATASET_RELEASE/in_repo/diversevul_dataset/splits/` (archive) | TODO: check redistribution |
| PrimeVul | Ding et al. 2024, authors' original default (unpaired) release | TODO: fetch step with URL and pinned sha256; until then place the three files in `$RERUN_ROOT/data/primevul/raw/` | TODO |
| SARD CWE-119 | DeepWukong authors' preprocessed archive | Manual download of `Data.7z` (see README), sha256 checked | TODO |

PrimeVul is used as released: every row, the official split, no subsampling. Its training sampler is the
class-balanced sampler also used for Big-Vul; validation and test keep their natural prevalence.
