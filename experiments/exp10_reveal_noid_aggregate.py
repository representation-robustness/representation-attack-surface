#!/usr/bin/env python3
"""Aggregate exp10 (REVEAL-NoId) 5-seed results into mean ± std JSON."""
import json, numpy as np
from pathlib import Path

DEVIGN = Path(__file__).resolve().parents[1] / "devign_full"
SEEDS  = [42, 1337, 7, 100, 999]
CONDS  = ["clean", "ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"]

per_seed = {}
for s in SEEDS:
    p = DEVIGN / f"reveal_noid_seed{s}_results.json"
    if p.exists():
        per_seed[s] = json.loads(p.read_text())
    else:
        print(f"  Missing: seed {s}")

if not per_seed:
    print("No seed results found."); raise SystemExit(1)

agg = {"n_seeds": len(per_seed), "seeds": list(per_seed.keys()),
       "model": "reveal_noid", "note": "structural-only REVEAL (Word2Vec zeroed)"}
for cond in CONDS:
    vals = [per_seed[s][cond]["f1"] for s in per_seed if cond in per_seed[s]
            and isinstance(per_seed[s][cond], dict) and "f1" in per_seed[s][cond]]
    if vals:
        agg[cond] = {"mean": round(float(np.mean(vals)), 2),
                     "std":  round(float(np.std(vals)), 2),
                     "vals": vals}
        clean_f1 = agg.get("clean", {}).get("mean", agg[cond]["mean"])
        print(f"  {cond}: {agg[cond]['mean']:.2f} ± {agg[cond]['std']:.2f}")

if "clean" in agg:
    print("\n  ΔF1 from clean:")
    for cond in ["ren", "dead", "cf", "compound"]:
        if cond in agg:
            delta = agg[cond]["mean"] - agg["clean"]["mean"]
            print(f"    Δ{cond}: {delta:+.2f}pp")

out = DEVIGN / "reveal_noid_multiseed_results.json"
out.write_text(json.dumps(agg, indent=2))
print(f"\nSaved → {out}")
