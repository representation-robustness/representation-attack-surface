# Shipped inputs

xz-compressed copies of the exact input files used in the paper. The `fetch` stage of `reproduce.sh` unpacks them
and checks each against `SHA256SUMS` (the same checksums as in `docs/DATASETS.md`).

| Directory | Dataset | Source | Licence |
|---|---|---|---|
| `bigvul/` | Big-Vul train/valid/test splits | Fan et al., "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries", MSR 2020 | MIT (upstream repository) |
| `diversevul/` | DiverseVul balanced train/valid/test splits | Chen et al., "DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection", RAID 2023 | No licence stated upstream; redistributed for research reproducibility, with attribution |
| `primevul/` | PrimeVul default (unpaired) release | Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?", ICSE 2025 | MIT (upstream repository) |

Please cite the original papers when using these files.
