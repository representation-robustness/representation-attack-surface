#!/usr/bin/env python3
"""Aggregate exp6 (ReGVD-Aug) 5-seed results into mean ± std JSON."""
import json, numpy as np
from pathlib import Path

DEVIGN = Path(__file__).resolve().parents[1] / "devign_full"
SEEDS  = [42, 1337, 7, 100, 999]
CONDS  = ["clean", "ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"]

per_seed = {}
for s in SEEDS:
    p = DEVIGN / f"regvd_aug_seed{s}_results.json"
    if p.exists():
        per_seed[s] = json.loads(p.read_text())
    else:
        print(f"  Missing: seed {s}")

if not per_seed:
    print("No seed results found."); raise SystemExit(1)

agg = {"n_seeds": len(per_seed), "seeds": list(per_seed.keys())}
for cond in CONDS:
    vals = [per_seed[s][cond]["f1"] for s in per_seed if cond in per_seed[s]]
    if vals:
        agg[cond] = {"mean": round(float(np.mean(vals)), 2),
                     "std":  round(float(np.std(vals)), 2),
                     "vals": vals}
        print(f"  {cond}: {agg[cond]['mean']:.2f} ± {agg[cond]['std']:.2f}")

out = DEVIGN / "regvd_aug_multiseed_results.json"
out.write_text(json.dumps(agg, indent=2))
print(f"\nSaved → {out}")
