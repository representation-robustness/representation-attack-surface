"""
aggregate_lmggnn_7cond.py — Aggregate per-seed LMGGNN eval results into 7cond summary files.

Usage:
    python utils/aggregate_lmggnn_7cond.py

Reads:  ~/thesis/devign_full/{dataset}_lmggnn_seed{seed}_results.json  (per-seed)
Writes: ~/thesis/devign_full/{dataset}_lmggnn_7cond_results.json
"""
import json
import os
import numpy as np

RESULT_DIR = os.path.expanduser("~/thesis/devign_full")
SEEDS = [42, 1337, 7, 100, 999]

SHORT_TO_CANONICAL = {
    "original":    "test",
    "identifier":  "test_obf_identifier",
    "deadcode":    "test_obf_deadcode",
    "controlflow": "test_obf_controlflow",
    "ren_dead":    "test_obf_ren_dead",
    "ren_cf":      "test_obf_ren_cf",
    "dead_cf":     "test_obf_dead_cf",
    "compound":    "test_obf_compound",
}


def aggregate_dataset(dataset):
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(RESULT_DIR, f"{dataset}_lmggnn_seed{seed}_results.json")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        per_seed.append(d)

    if not per_seed:
        print(f"No results found for {dataset}")
        return

    # Collect all conditions that appear in any seed result
    all_conds = set()
    for r in per_seed:
        all_conds.update(r.keys())
    all_conds = [c for c in SHORT_TO_CANONICAL if c in all_conds]

    agg = {}
    for cond in all_conds:
        f1s = [r[cond]["f1"] for r in per_seed if cond in r]
        if not f1s:
            continue
        agg[SHORT_TO_CANONICAL[cond]] = {
            "f1_mean": round(float(np.mean(f1s)), 2),
            "f1_std":  round(float(np.std(f1s)), 2),
            "all_f1":  f1s,
        }

    if "test" in agg:
        base = agg["test"]["f1_mean"]
        for can_key in agg:
            if can_key != "test":
                agg[can_key]["delta_f1"] = round(agg[can_key]["f1_mean"] - base, 2)

    agg["n_seeds"] = len(per_seed)
    agg["seeds"]   = [SEEDS[i] for i, _ in enumerate(per_seed)]
    agg["model"]   = "Vul-LMGGNN"
    agg["dataset"] = dataset

    out = os.path.join(RESULT_DIR, f"{dataset}_lmggnn_7cond_results.json")
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n{dataset} → {out}")
    for k, v in agg.items():
        if isinstance(v, dict):
            delta = f"  Δ={v.get('delta_f1', 0):+.2f}" if k != "test" else ""
            print(f"  {k:<35} F1={v['f1_mean']:.2f}±{v['f1_std']:.2f}%{delta}")


if __name__ == "__main__":
    for ds in ["bigvul", "diversevul"]:
        print(f"\n=== {ds} ===")
        aggregate_dataset(ds)
