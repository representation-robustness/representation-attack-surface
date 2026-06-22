#!/usr/bin/env python3
"""
Exp 1: Bootstrap confidence intervals for all robustness results.

Two levels:
  - Seed-level bootstrap (all models): resample from 5 seed F1 values
  - Function-level bootstrap (models with per-function preds): resample test functions

Output: devign_full/bootstrap_ci_results.json
"""

import json, os, sys
import numpy as np
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
DEVIGN = THESIS / "devign_full"
PREDS  = DEVIGN / "attack" / "preds"
OUT    = DEVIGN / "bootstrap_ci_results.json"

N_BOOT = 10000
RNG    = np.random.default_rng(42)

def boot_ci_mean(vals, n=N_BOOT, ci=95):
    """Bootstrap CI for the mean of a list of values."""
    vals = np.array(vals, dtype=float)
    if len(vals) < 2:
        return float(vals[0]) if len(vals) == 1 else None, None, None
    boot = RNG.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, (100-ci)/2))
    hi = float(np.percentile(boot, 100 - (100-ci)/2))
    return round(float(vals.mean()), 4), round(lo, 4), round(hi, 4)

def boot_ci_delta_f1(clean_preds, obf_preds, true_labels, n=N_BOOT, ci=95):
    """Bootstrap CI for delta-F1 by resampling test functions."""
    from sklearn.metrics import f1_score
    c  = np.array(clean_preds)
    o  = np.array(obf_preds)
    y  = np.array(true_labels)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        f_clean = f1_score(y[s], c[s], zero_division=0) * 100
        f_obf   = f1_score(y[s], o[s], zero_division=0) * 100
        deltas.append(f_obf - f_clean)
    lo = float(np.percentile(deltas, (100-ci)/2))
    hi = float(np.percentile(deltas, 100 - (100-ci)/2))
    return round(lo, 4), round(hi, 4)

# ── Seed-level bootstrap ─────────────────────────────────────────────────────
MULTISEED_FILES = {
    # Devign
    "devign_ecgrgcn":   DEVIGN / "ecgrgcn_multiseed_results.json",
    "devign_angle":     DEVIGN / "angle_multiseed_results.json",
    "devign_vulgnn":    DEVIGN / "vulgnn_multiseed_results.json",
    "devign_reveal":    DEVIGN / "reveal_7cond_results.json",
    "devign_regvd":     DEVIGN / "regvd_multiseed_results.json",
    "devign_lmggnn":    DEVIGN / "lmggnn_multiseed_results.json",
    "devign_codebert":  DEVIGN / "codebert_multiseed_results.json",
    "devign_codet5":    DEVIGN / "codet5plus_multiseed_results.json",
    # BigVul
    "bigvul_ecgrgcn":   DEVIGN / "bigvul_ecgrgcn_multiseed_results.json",
    "bigvul_angle":     DEVIGN / "bigvul_angle_multiseed_results.json",
    "bigvul_vulgnn":    DEVIGN / "bigvul_vulgnn_multiseed_results.json",
    "bigvul_reveal":    DEVIGN / "bigvul_reveal_multiseed_results.json",
    "bigvul_regvd":     DEVIGN / "bigvul_regvd_multiseed_results.json",
    "bigvul_lmggnn":    DEVIGN / "bigvul_lmggnn_multiseed_results.json",
    "bigvul_codebert":  DEVIGN / "bigvul_codebert_multiseed_results.json",
    "bigvul_codet5":    DEVIGN / "bigvul_codet5plus_multiseed_results.json",
    # DiverseVul
    "dv_ecgrgcn":       DEVIGN / "diversevul_ecgrgcn_multiseed_results.json",
    "dv_angle":         DEVIGN / "diversevul_angle_multiseed_results.json",
    "dv_vulgnn":        DEVIGN / "diversevul_vulgnn_multiseed_results.json",
    "dv_reveal":        DEVIGN / "diversevul_reveal_multiseed_results.json",
    "dv_regvd":         DEVIGN / "diversevul_regvd_multiseed_results.json",
    "dv_lmggnn":        DEVIGN / "diversevul_lmggnn_multiseed_results.json",
    "dv_codebert":      DEVIGN / "diversevul_codebert_multiseed_results.json",
    "dv_codet5":        DEVIGN / "diversevul_codet5plus_multiseed_results.json",
}

COND_MAP = {
    "test":                  "clean",
    "test_obf_identifier":   "ren",
    "test_obf_deadcode":     "dead",
    "test_obf_controlflow":  "cf",
    "test_obf_compound":     "compound",
    # 7-cond keys
    "original":   "clean",
    "identifier": "ren",
    "deadcode":   "dead",
    "controlflow":"cf",
    "ren_dead":   "ren_dead",
    "ren_cf":     "ren_cf",
    "dead_cf":    "dead_cf",
}

results = {"seed_bootstrap": {}, "function_bootstrap": {}}

print("=== Seed-level bootstrap ===", flush=True)
for name, path in MULTISEED_FILES.items():
    if not path.exists():
        print(f"  SKIP {name} (file not found)", flush=True)
        continue
    d = json.loads(path.read_text())

    # collect per-condition F1 arrays
    model_ci = {}
    for raw_key, cond in COND_MAP.items():
        if raw_key not in d:
            continue
        cond_data = d[raw_key]
        f1_vals = cond_data.get("all_f1")
        if not f1_vals:
            continue
        mean, lo, hi = boot_ci_mean(f1_vals)
        model_ci[cond] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi,
                          "n_seeds": len(f1_vals)}
        # delta F1 CI (vs clean)
        if cond != "clean" and "clean" in model_ci:
            clean_vals = []
            for ck in COND_MAP:
                if COND_MAP[ck] == "clean" and ck in d:
                    clean_vals = d[ck].get("all_f1", [])
                    break
            obf_vals = f1_vals
            if len(clean_vals) == len(obf_vals) and len(clean_vals) > 1:
                delta_vals = [o - c for c, o in zip(clean_vals, obf_vals)]
                dmean, dlo, dhi = boot_ci_mean(delta_vals)
                model_ci[cond]["delta_mean"] = dmean
                model_ci[cond]["delta_ci95_lo"] = dlo
                model_ci[cond]["delta_ci95_hi"] = dhi

    results["seed_bootstrap"][name] = model_ci
    print(f"  {name}: {list(model_ci.keys())}", flush=True)

# ── Function-level bootstrap ─────────────────────────────────────────────────
print("\n=== Function-level bootstrap (per-function preds) ===", flush=True)
try:
    from sklearn.metrics import f1_score
    SKIP_IMPORT = False
except ImportError:
    print("  sklearn not available, skipping function-level bootstrap", flush=True)
    SKIP_IMPORT = True

if not SKIP_IMPORT and PREDS.exists():
    for pred_file in sorted(PREDS.glob("*_preds.json")):
        d = json.loads(pred_file.read_text())
        model_name = d["model"]
        true_labels = d["true_labels"]
        seeds = d.get("seeds", {})
        conds = d.get("conditions", [])
        func_ci = {}

        for seed_key, cond_preds in seeds.items():
            clean_p = cond_preds.get("clean", [])
            n = min(len(true_labels), len(clean_p))
            for cond in conds:
                if cond == "clean":
                    continue
                obf_p = cond_preds.get(cond, [])
                if len(obf_p) < n:
                    continue
                lo, hi = boot_ci_delta_f1(clean_p[:n], obf_p[:n], true_labels[:n])
                key = f"seed{seed_key}_{cond}"
                func_ci[key] = {"delta_ci95_lo": lo, "delta_ci95_hi": hi}

        results["function_bootstrap"][model_name] = func_ci
        print(f"  {model_name}: {len(func_ci)} condition×seed CIs computed", flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {OUT}", flush=True)
