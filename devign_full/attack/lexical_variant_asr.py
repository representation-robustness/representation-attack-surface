#!/usr/bin/env python3
"""
Compute per-variant ASR for the three lexical attack types on token-aware models.

Outputs:
  attack/lexical_variant_asr.json   -- structured results
  attack/lexical_variant_table.txt  -- LaTeX-ready table rows

Lexical variants tested:
  ren          -- uniform renaming (original)
  vocab_shift  -- OOV-style renaming
  benign_token -- safety-suggesting renaming

Models included (token-aware only):
  tfidf, codebert, regvd, codet5plus, reveal
  (structural-only models excluded — renaming-invariant)
"""

import json
import numpy as np
from pathlib import Path

PREDS_DIR = Path(__file__).parent / "preds"
OUT_DIR   = Path(__file__).parent

LEXICAL_VARIANTS = ["ren", "vocab_shift", "benign_token"]

# Models that are token-aware (exposed to identifier tokens)
TOKEN_AWARE_MODELS = {
    "reveal":    "REVEAL",
    "regvd":     "ReGVD",
    "codebert":  "CodeBERT",
    "codet5plus":"CodeT5+",
    "tfidf":     "TF-IDF+LR",
}


def compute_variant_asr(preds_file: Path):
    data = json.load(open(preds_file))
    model_name  = data["model"]
    true_labels = data["true_labels"]
    seeds       = data["seeds"]

    n = len(true_labels)
    results = {}

    for seed_key, cond_preds in seeds.items():
        clean_preds = cond_preds.get("clean", [])
        if not clean_preds:
            continue
        min_n = min(n, len(clean_preds))

        # Correctly-classified vulnerable functions
        vuln_correct_idx = [
            i for i in range(min_n)
            if true_labels[i] == 1 and clean_preds[i] == 1
        ]
        if not vuln_correct_idx:
            continue

        for variant in LEXICAL_VARIANTS:
            if variant not in cond_preds:
                continue
            vpreds = cond_preds[variant][:min_n]
            evaded = sum(1 for i in vuln_correct_idx if vpreds[i] == 0)
            asr    = evaded / len(vuln_correct_idx) * 100

            if variant not in results:
                results[variant] = []
            results[variant].append(asr)

    return {
        "model": model_name,
        "n_seeds": len(seeds),
        "variants": {
            v: {
                "mean": round(float(np.mean(vals)), 1),
                "std":  round(float(np.std(vals)), 1),
            }
            for v, vals in results.items()
            if vals
        }
    }


def main():
    all_results = {}
    for fname in PREDS_DIR.glob("*_preds.json"):
        name = fname.stem.replace("_preds", "")
        if name not in TOKEN_AWARE_MODELS:
            continue
        r = compute_variant_asr(fname)
        if r and r["variants"]:
            all_results[name] = r
            print(f"\n{TOKEN_AWARE_MODELS[name]}:")
            for v, stats in r["variants"].items():
                print(f"  {v:15s}: ASR={stats['mean']:.1f}±{stats['std']:.1f}%")

    # Save JSON
    out_json = OUT_DIR / "lexical_variant_asr.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json}", flush=True)

    # LaTeX table
    print("\n=== LaTeX table rows ===")
    print(r"\midrule")
    MODEL_ORDER = ["reveal", "regvd", "codebert", "codet5plus", "tfidf"]
    for name in MODEL_ORDER:
        if name not in all_results:
            print(f"% {TOKEN_AWARE_MODELS.get(name, name)}: no data")
            continue
        r  = all_results[name]
        vr = r["variants"]
        ren_s  = f"{vr['ren']['mean']:.1f}"   if "ren"          in vr else "---"
        vsh_s  = f"{vr['vocab_shift']['mean']:.1f}" if "vocab_shift"  in vr else "---"
        ben_s  = f"{vr['benign_token']['mean']:.1f}" if "benign_token" in vr else "---"
        label  = TOKEN_AWARE_MODELS.get(name, name)
        print(f"{label} & {ren_s} & {vsh_s} & {ben_s} \\\\")


if __name__ == "__main__":
    main()
