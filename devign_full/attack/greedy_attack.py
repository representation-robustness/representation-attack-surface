#!/usr/bin/env python3
"""
Greedy minimum-budget source-level evasion attack.

For each correctly-classified vulnerable test function, finds the minimum
number of semantics-preserving transforms needed to evade detection:
  Budget 1: try Ren, Dead, CF individually
  Budget 2: try R+D, R+CF, D+CF (pairwise)
  Budget 3: try Compound (all three)

For renaming-invariant models (ECG RGCN, VulGNN): pairwise shortcuts applied
  R+D = Dead,  R+CF = CF,  D+CF = Compound

For REVEAL: only budget-1 analysis (pairwise embeddings not available).

Outputs:
  attack/greedy_attack_results.json
  attack/greedy_attack_table.txt  (LaTeX-ready)
"""

import json
import numpy as np
from pathlib import Path

PREDS_DIR = Path(__file__).parent / "preds"
OUT_DIR   = Path(__file__).parent

RENAMING_INVARIANT = {"ecg_rgcn", "vulgnn"}

BUDGET1_CONDS = ["ren", "dead", "cf"]
BUDGET2_CONDS = ["ren_dead", "ren_cf", "dead_cf"]
BUDGET3_CONDS = ["compound"]

# For renaming-invariant models, map pairwise to single equivalents
INVARIANCE_MAP = {"ren_dead": "dead", "ren_cf": "cf", "dead_cf": "compound"}


def compute_asr(preds_file: Path):
    data = json.load(open(preds_file))
    model_name   = data["model"]
    true_labels  = data["true_labels"]
    seeds        = data["seeds"]
    renaming_inv = model_name in RENAMING_INVARIANT
    avail_conds  = data.get("conditions", list(next(iter(seeds.values())).keys()))

    per_seed_stats = []

    for seed_key, cond_preds in seeds.items():
        clean_preds = cond_preds.get("clean", [])
        # truncate to minimum length across all conditions to handle parser-failure count mismatches
        n = min(len(true_labels), len(clean_preds),
                min((len(v) for v in cond_preds.values()), default=len(clean_preds)))
        if n == 0:
            continue
        true_labels_n = true_labels[:n]

        # correctly-classified vulnerable functions
        vuln_correct_idx = [
            i for i in range(n)
            if true_labels_n[i] == 1 and clean_preds[i] == 1
        ]
        total_vuln_correct = len(vuln_correct_idx)
        if total_vuln_correct == 0:
            continue

        def is_evaded_by(cond):
            if cond in cond_preds:
                return cond_preds[cond][:n]
            if renaming_inv and cond in INVARIANCE_MAP:
                proxy = INVARIANCE_MAP[cond]
                return cond_preds.get(proxy, [1] * n)[:n]
            return [1] * n   # not available → assume not evaded

        evaded_b1 = set()
        for cond in BUDGET1_CONDS:
            preds = is_evaded_by(cond)
            for i in vuln_correct_idx:
                if preds[i] == 0:
                    evaded_b1.add(i)

        evaded_b2 = set(evaded_b1)
        for cond in BUDGET2_CONDS:
            preds = is_evaded_by(cond)
            for i in vuln_correct_idx:
                if preds[i] == 0:
                    evaded_b2.add(i)

        evaded_b3 = set(evaded_b2)
        for cond in BUDGET3_CONDS:
            preds = is_evaded_by(cond)
            for i in vuln_correct_idx:
                if preds[i] == 0:
                    evaded_b3.add(i)

        # min budget per function
        min_budgets = []
        for i in vuln_correct_idx:
            if i in evaded_b1:
                min_budgets.append(1)
            elif i in evaded_b2:
                min_budgets.append(2)
            elif i in evaded_b3:
                min_budgets.append(3)
            else:
                min_budgets.append(None)

        per_seed_stats.append({
            "total_vuln_correct": total_vuln_correct,
            "asr_b1": len(evaded_b1) / total_vuln_correct * 100,
            "asr_b2": len(evaded_b2) / total_vuln_correct * 100,
            "asr_b3": len(evaded_b3) / total_vuln_correct * 100,
            "mean_min_budget": np.mean([b for b in min_budgets if b is not None])
                               if any(b is not None for b in min_budgets) else None,
        })

    if not per_seed_stats:
        return None

    return {
        "model": model_name,
        "n_seeds": len(per_seed_stats),
        "renaming_invariant": renaming_inv,
        "total_vuln_correct": round(np.mean([s["total_vuln_correct"] for s in per_seed_stats]), 1),
        "asr_b1_mean": round(np.mean([s["asr_b1"] for s in per_seed_stats]), 1),
        "asr_b1_std":  round(np.std( [s["asr_b1"] for s in per_seed_stats]), 1),
        "asr_b2_mean": round(np.mean([s["asr_b2"] for s in per_seed_stats]), 1),
        "asr_b2_std":  round(np.std( [s["asr_b2"] for s in per_seed_stats]), 1),
        "asr_b3_mean": round(np.mean([s["asr_b3"] for s in per_seed_stats]), 1),
        "asr_b3_std":  round(np.std( [s["asr_b3"] for s in per_seed_stats]), 1),
        "mean_min_budget": round(np.mean([s["mean_min_budget"] for s in per_seed_stats
                                          if s["mean_min_budget"] is not None]), 2),
    }


MODEL_ORDER = ["ecg_rgcn", "angle", "vulgnn", "reveal", "regvd", "codebert", "codet5plus", "tfidf"]
MODEL_LABELS = {
    "ecg_rgcn":  "ECG RGCN",
    "angle":     "ANGLE",
    "vulgnn":    "VulGNN",
    "reveal":    "REVEAL",
    "regvd":     "ReGVD",
    "codebert":  "CodeBERT",
    "codet5plus":"CodeT5+",
    "tfidf":     "TF-IDF+LR",
}


def main():
    results = {}
    for fname in PREDS_DIR.glob("*_preds.json"):
        name = fname.stem.replace("_preds", "")
        r = compute_asr(fname)
        if r:
            results[name] = r
            print(f"{name}: ASR@1={r['asr_b1_mean']:.1f}±{r['asr_b1_std']:.1f}%  "
                  f"ASR@2={r['asr_b2_mean']:.1f}%  ASR@3={r['asr_b3_mean']:.1f}%  "
                  f"minBudget={r['mean_min_budget']:.2f}", flush=True)

    # save JSON
    out_json = OUT_DIR / "greedy_attack_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}", flush=True)

    # print LaTeX-ready table rows
    print("\n=== LaTeX table rows ===")
    print(r"\midrule")
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        label = MODEL_LABELS.get(name, name)
        inv   = "\\checkmark" if r["renaming_invariant"] else ""
        b1    = f"{r['asr_b1_mean']:.1f}"
        b2    = f"{r['asr_b2_mean']:.1f}" if r["asr_b2_mean"] != r["asr_b1_mean"] else "—"
        b3    = f"{r['asr_b3_mean']:.1f}" if r["asr_b3_mean"] != r["asr_b2_mean"] else "—"
        mb    = f"{r['mean_min_budget']:.2f}" if r['mean_min_budget'] else "—"
        print(f"{label} & {inv} & {b1} & {b2} & {b3} & {mb} \\\\")

    # plain text summary
    print("\n=== Summary ===")
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        print(f"  {MODEL_LABELS.get(name, name):15s}  "
              f"ASR@budget1={r['asr_b1_mean']:5.1f}%  "
              f"ASR@budget2={r['asr_b2_mean']:5.1f}%  "
              f"ASR@budget3={r['asr_b3_mean']:5.1f}%  "
              f"minBudget={r['mean_min_budget']:.2f}")


if __name__ == "__main__":
    main()
