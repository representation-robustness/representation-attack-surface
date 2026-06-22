#!/usr/bin/env python3
"""
Exp 5: Compute transfer attack success rate (ASR) matrix across detectors.

For each source model, identifies the single best-budget-1 transform (the one
that evades the source most), then measures what fraction of those same functions
are also evaded by each target model under the same transform.

Run after Exp 3 and Exp 4 have saved prediction files.

Output: devign_full/transfer_asr_matrix.json
"""

import json, sys
import numpy as np
from pathlib import Path

THESIS   = Path(__file__).resolve().parents[1]
DEVIGN   = THESIS / "devign_full"
PREDS    = DEVIGN / "attack" / "preds"
OUT_FILE = DEVIGN / "transfer_asr_matrix.json"

SINGLE_TRANSFORMS = ["ren", "dead", "cf"]

# Which transforms each source "selects" (best budget-1 attack per representation type)
# Lexical sources → best transform is usually "ren"
# Structural sources → best transform is usually "dead" or "cf"
LEXICAL_MODELS    = {"reveal", "regvd", "codebert", "codet5plus", "tfidf"}
STRUCTURAL_MODELS = {"angle", "vulgnn", "ecg_rgcn"}

def load_preds(path):
    d = json.loads(Path(path).read_text())
    return d

def get_best_budget1_transform(preds_data, true_labels, N):
    """Return the single-transform condition with highest ASR for this model."""
    seeds   = preds_data.get("seeds", {})
    best_t, best_asr = None, -1
    for cond in SINGLE_TRANSFORMS:
        asrs = []
        for seed_preds in seeds.values():
            clean_p = seed_preds.get("clean", [])
            obf_p   = seed_preds.get(cond, [])
            n = min(N, len(clean_p), len(obf_p))
            if n == 0:
                continue
            # correctly classified vulnerable functions
            vc = [i for i in range(n) if true_labels[i] == 1 and clean_p[i] == 1]
            if not vc:
                continue
            evaded = sum(1 for i in vc if obf_p[i] == 0)
            asrs.append(evaded / len(vc) * 100)
        if asrs:
            mean_asr = np.mean(asrs)
            if mean_asr > best_asr:
                best_asr = mean_asr
                best_t   = cond
    return best_t, best_asr

def transfer_asr(attack_transform, source_data, target_data, true_labels, N):
    """
    ASR on target when using attack_transform selected from source.
    Computed over functions correctly classified by BOTH source and target.
    """
    s_seeds = source_data.get("seeds", {})
    t_seeds = target_data.get("seeds", {})
    all_asr = []
    for s_seed_preds in s_seeds.values():
        s_clean = s_seed_preds.get("clean", [])
        s_obf   = s_seed_preds.get(attack_transform, [])
        for t_seed_preds in t_seeds.values():
            t_clean = t_seed_preds.get("clean", [])
            t_obf   = t_seed_preds.get(attack_transform, [])
            n = min(N, len(s_clean), len(t_clean), len(s_obf), len(t_obf))
            if n == 0:
                continue
            # both models correctly classify the function as vulnerable (clean)
            both_correct = [
                i for i in range(n)
                if true_labels[i] == 1 and s_clean[i] == 1 and t_clean[i] == 1
            ]
            if not both_correct:
                continue
            # evaded on target by source's chosen transform
            evaded = sum(1 for i in both_correct if t_obf[i] == 0)
            all_asr.append(evaded / len(both_correct) * 100)
    return round(float(np.mean(all_asr)), 2) if all_asr else None

# Load all prediction files
pred_files = sorted(PREDS.glob("*_preds.json"))
if not pred_files:
    print(f"No pred files found in {PREDS}. Run Exp3 and Exp4 first.", flush=True)
    sys.exit(1)

all_preds = {}
for pf in pred_files:
    d = load_preds(pf)
    all_preds[d["model"]] = d
    print(f"Loaded: {d['model']} ({len(d.get('seeds',{}))} seeds, "
          f"conds: {d.get('conditions',[])})", flush=True)

# Get true labels (should be consistent across files)
ref_model  = list(all_preds.values())[0]
true_labels = ref_model["true_labels"]
N           = len(true_labels)
print(f"\nTrue labels: {N} functions", flush=True)

# Source models for transfer table (matching paper Table 10)
SOURCE_MODELS  = ["reveal", "regvd", "codebert", "angle", "vulgnn"]
TARGET_MODELS  = ["reveal", "regvd", "codebert", "codet5plus", "angle", "vulgnn"]

matrix  = {}
details = {}

for src in SOURCE_MODELS:
    if src not in all_preds:
        print(f"Source {src}: predictions not found, skipping", flush=True)
        continue
    src_data  = all_preds[src]
    best_t, best_asr_self = get_best_budget1_transform(src_data, true_labels, N)
    if best_t is None:
        print(f"Source {src}: could not determine best transform", flush=True)
        continue

    print(f"\nSource: {src}  best_transform={best_t}  self_ASR={best_asr_self:.1f}%", flush=True)
    row = {"best_transform": best_t, "self_asr": round(best_asr_self, 2), "targets": {}}

    for tgt in TARGET_MODELS:
        if tgt not in all_preds:
            row["targets"][tgt] = None
            continue
        if tgt == src:
            row["targets"][tgt] = round(best_asr_self, 2)
            continue
        asr = transfer_asr(best_t, src_data, all_preds[tgt], true_labels, N)
        row["targets"][tgt] = asr
        print(f"  → {tgt}: ASR={asr}%", flush=True)

    matrix[src] = row

# Also compute: for each transform, aggregate ASR on every model
print("\n=== Per-transform aggregate ASR ===", flush=True)
transform_asr = {}
for t in SINGLE_TRANSFORMS:
    t_row = {}
    for model_name, preds_data in all_preds.items():
        seeds = preds_data.get("seeds", {})
        asrs  = []
        for seed_preds in seeds.values():
            clean_p = seed_preds.get("clean", [])
            obf_p   = seed_preds.get(t, [])
            n = min(N, len(clean_p), len(obf_p))
            vc = [i for i in range(n) if true_labels[i] == 1 and clean_p[i] == 1]
            if not vc:
                continue
            evaded = sum(1 for i in vc if obf_p[i] == 0)
            asrs.append(evaded / len(vc) * 100)
        t_row[model_name] = round(float(np.mean(asrs)), 2) if asrs else None
    transform_asr[t] = t_row
    print(f"  {t}: {t_row}", flush=True)

out = {
    "transfer_matrix": matrix,
    "per_transform_asr": transform_asr,
    "n_test_functions": N,
}
OUT_FILE.write_text(json.dumps(out, indent=2))
print(f"\nSaved → {OUT_FILE}", flush=True)
