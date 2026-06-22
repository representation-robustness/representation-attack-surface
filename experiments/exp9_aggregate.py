#!/usr/bin/env python3
"""
Aggregate CodeBERT-Curriculum results across 5 seeds into a multiseed JSON.
Output matches the format produced by exp8_codebert_mixed_aggregate.py.

Usage:
    python experiments/exp9_aggregate.py
"""

import json
import numpy as np
from pathlib import Path

THESIS  = Path(__file__).resolve().parents[1]
DEVIGN  = THESIS / "devign_full"
SEEDS   = [42, 1337, 7, 100, 999]
CONDITIONS = ["clean", "ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"]

def main():
    per_seed = {}
    missing  = []
    for s in SEEDS:
        p = DEVIGN / f"codebert_curriculum_seed{s}_results.json"
        if not p.exists():
            missing.append(s); continue
        per_seed[s] = json.loads(p.read_text())

    if missing:
        print(f"Missing seeds: {missing}")
        if len(per_seed) == 0:
            raise SystemExit("No seed results found — run exp9 first.")

    # Aggregate per condition
    agg = {}
    for cond in CONDITIONS:
        vals = [per_seed[s][cond]["f1"] for s in per_seed if cond in per_seed[s]]
        if not vals:
            continue
        agg[cond] = {
            "mean": round(float(np.mean(vals)), 2),
            "std":  round(float(np.std(vals)),  2),
            "per_seed": {str(s): per_seed[s][cond]["f1"] for s in per_seed if cond in per_seed[s]},
        }

    # ΔF1 relative to clean
    clean_mean = agg.get("clean", {}).get("mean", None)
    if clean_mean is not None:
        for cond in CONDITIONS:
            if cond == "clean" or cond not in agg:
                continue
            agg[cond]["delta"] = round(agg[cond]["mean"] - clean_mean, 2)

    # Max |Δ| over single-transform conditions
    single = ["ren", "dead", "cf"]
    deltas = [abs(agg[c]["delta"]) for c in single if c in agg and "delta" in agg[c]]
    max_delta = round(max(deltas), 2) if deltas else None

    out = {
        "model": "codebert_curriculum",
        "training": "curriculum",
        "seeds": SEEDS,
        "n_seeds": len(per_seed),
        "conditions": agg,
        "max_abs_delta_single": max_delta,
    }

    out_path = DEVIGN / "codebert_curriculum_multiseed_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved → {out_path}")

    # Print summary table
    print(f"\nCodeBERT-Curriculum  ({len(per_seed)}/{len(SEEDS)} seeds)")
    print(f"  Clean F1:  {agg['clean']['mean']:.2f} ± {agg['clean']['std']:.2f}")
    for cond in ["ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"]:
        if cond not in agg:
            continue
        d = agg[cond].get("delta", "?")
        sign = "+" if isinstance(d, float) and d >= 0 else ""
        print(f"  Δ{cond:<8s}: {sign}{d}")
    print(f"  Max|Δ| (single): {max_delta}")

    # LaTeX row
    clean_str = f"{agg['clean']['mean']:.2f}\\,\\pm\\,{agg['clean']['std']:.2f}"
    cols = []
    for cond in ["ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"]:
        if cond not in agg or "delta" not in agg[cond]:
            cols.append("---")
        else:
            d = agg[cond]["delta"]
            cols.append(f"${'+' if d >= 0 else ''}{d:.2f}$")
    cols.append(str(max_delta) if max_delta is not None else "---")
    print(f"\nLaTeX row (Trans family):")
    print(f" & CodeBERT-Cur$^*$$\\ddagger$ & {clean_str} & {' & '.join(cols)} \\\\")


if __name__ == "__main__":
    main()
