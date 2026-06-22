"""
Bootstrap CIs over trials/seeds for all Devign tab:devign models.
Outputs per-model table of ΔF1 values and whether each CI excludes zero.
"""
import json, numpy as np, os

np.random.seed(0)
N_BOOT = 10_000

def bootstrap_ci(deltas, alpha=0.05):
    """95% CI for mean delta via percentile bootstrap over trials."""
    arr = np.array(deltas, dtype=float)
    n = len(arr)
    boot_means = [np.mean(arr[np.random.randint(0, n, n)]) for _ in range(N_BOOT)]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi

def reliable(lo, hi):
    """Returns True if 95% CI excludes zero (both same sign)."""
    return lo > 0 or hi < 0

# ── model specs ──────────────────────────────────────────────────────────────
# Each entry: (display_name, file, clean_key, cond_keys)
MODELS_DEVIGN = [
    ("ECG RGCN",    "devign_full/ecgrgcn_7cond_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("ANGLE",       "devign_full/angle_7cond_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("VulGNN",      "devign_full/vulgnn_7cond_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("REVEAL",      "devign_full/reveal_7cond_results.json",
     "originals", {"Ren":"identifier","Dead":"deadcode","CF":"controlflow",
                    "R+D":"ren_dead","R+CF":"ren_cf","D+CF":"dead_cf","Cmp":"compound"}),
    ("Dev.GGNN",    "devign_full/devign_ggnn_7cond_results.json",
     "original", {"Ren":"identifier","Dead":"deadcode","CF":"controlflow",
                   "R+D":"ren_dead","R+CF":"ren_cf","D+CF":"dead_cf","Cmp":"compound"}),
    ("LMGGNN",      "devign_full/lmggnn_7cond_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("CodeT5+",     "devign_full/codet5_7cond_devign_results.json",
     "original", {"Ren":"identifier","Dead":"deadcode","CF":"controlflow",
                   "R+D":"ren_dead","R+CF":"ren_cf","D+CF":"dead_cf","Cmp":"compound"}),
    ("TF-IDF+LR",   "devign_full/tfidf_7cond_devign_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("CPG+LR",      "devign_full/cpglr_7cond_devign_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("CodeBERT-Aug","devign_full/codebert_aug_7cond_results.json",
     "test", {"Ren":"test_obf_identifier","Dead":"test_obf_deadcode","CF":"test_obf_controlflow",
               "R+D":"test_obf_ren_dead","R+CF":"test_obf_ren_cf","D+CF":"test_obf_dead_cf","Cmp":"test_obf_compound"}),
    ("CodeBERT-Cur","devign_full/codebert_curriculum_multiseed_results.json",
     None, None),  # handled specially
]

CONDS = ["Ren","Dead","CF","R+D","R+CF","D+CF","Cmp"]

results = {}

for name, fpath, clean_key, cond_map in MODELS_DEVIGN:
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue
    d = json.load(open(fpath))

    if name == "CodeBERT-Cur":
        # curriculum multiseed: d["conditions"][key]["per_seed"] dict
        conds_d = d["conditions"]
        seeds = [str(s) for s in d["seeds"]]
        clean_f1 = np.array([conds_d["clean"]["per_seed"][s] for s in seeds], dtype=float)
        row = {}
        for label, key in [("Ren","ren"),("Dead","dead"),("CF","cf"),
                            ("R+D","ren_dead"),("R+CF","ren_cf"),("D+CF","dead_cf"),("Cmp","compound")]:
            cond_f1 = np.array([conds_d[key]["per_seed"][s] for s in seeds], dtype=float)
            deltas = cond_f1 - clean_f1
            lo, hi = bootstrap_ci(deltas)
            row[label] = {"mean": float(np.mean(deltas)), "lo": lo, "hi": hi, "reliable": reliable(lo, hi)}
        results[name] = row
        continue

    clean_all = d[clean_key]["all_f1"]
    row = {}
    for label, ckey in cond_map.items():
        if ckey not in d:
            row[label] = None
            continue
        cond_all = d[ckey]["all_f1"]
        if len(cond_all) != len(clean_all):
            print(f"  WARNING: {name} {label}: len mismatch {len(clean_all)} vs {len(cond_all)}")
        deltas = [c - cl for c, cl in zip(cond_all, clean_all)]
        lo, hi = bootstrap_ci(deltas)
        row[label] = {"mean": float(np.mean(deltas)), "lo": lo, "hi": hi, "reliable": reliable(lo, hi)}
    results[name] = row

# ── print summary ─────────────────────────────────────────────────────────────
print(f"\n{'Model':<15} {'':>5} {'Ren':>6} {'Dead':>6} {'CF':>6} {'R+D':>6} {'R+CF':>6} {'D+CF':>6} {'Cmp':>6}")
print("-" * 70)
for name in results:
    row = results[name]
    parts = []
    for c in CONDS:
        v = row.get(c)
        if v is None:
            parts.append("  N/A  ")
        else:
            star = "*" if v["reliable"] else " "
            parts.append(f"{v['mean']:+5.2f}{star}")
    print(f"{name:<15} " + " ".join(parts))

# ── save full results ─────────────────────────────────────────────────────────
out = {}
for name, row in results.items():
    out[name] = {}
    for cond, v in row.items():
        if v is not None:
            out[name][cond] = {
                "delta_mean": round(v["mean"], 3),
                "ci95_lo": round(v["lo"], 3),
                "ci95_hi": round(v["hi"], 3),
                "reliable": bool(v["reliable"])
            }

with open("devign_full/bootstrap_ci_v2.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to devign_full/bootstrap_ci_v2.json")
