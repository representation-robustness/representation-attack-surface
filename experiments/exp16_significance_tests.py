"""
Paired one-sample t-tests (H0: mean delta = 0) with per-table Bonferroni correction.
For each model × condition, tests whether the transformation significantly shifts F1.
Covers all 5 tables: Devign, BigVul, DiverseVul, MitigCodeBERT, MitigAll.
"""
import json, numpy as np, os
from scipy import stats

CONDS = ["Ren", "Dead", "CF", "R+D", "R+CF", "D+CF", "Cmp"]

STD_COND_MAP = {c: k for c, k in zip(CONDS, [
    "test_obf_identifier", "test_obf_deadcode", "test_obf_controlflow",
    "test_obf_ren_dead",   "test_obf_ren_cf",   "test_obf_dead_cf",
    "test_obf_compound"
])}

REVEAL_DEVIGN_MAP = {c: k for c, k in zip(CONDS, [
    "identifier", "deadcode", "controlflow",
    "ren_dead", "ren_cf", "dead_cf", "compound"
])}

# ── data loaders ──────────────────────────────────────────────────────────────

def from_7cond(fpath, clean_key, cond_map):
    """Per-seed deltas from an aggregated 7-cond file (all_f1 arrays)."""
    d = json.load(open(fpath))
    if clean_key not in d or not isinstance(d[clean_key], dict):
        return {}
    clean_f1 = d[clean_key].get("all_f1", [])
    if len(clean_f1) < 2:
        return {}  # not enough data for t-test
    result = {}
    for label, ckey in cond_map.items():
        if ckey not in d or not isinstance(d[ckey], dict):
            continue
        cond_f1 = d[ckey].get("all_f1", [])
        n = min(len(clean_f1), len(cond_f1))
        if n >= 2:
            result[label] = [cond_f1[i] - clean_f1[i] for i in range(n)]
    return result

def from_multiseed_vals(fpath):
    """Per-seed deltas from aug-style multiseed file (vals arrays)."""
    d = json.load(open(fpath))
    clean_vals = d["clean"]["vals"]
    result = {}
    for label, key in zip(CONDS, ["ren","dead","cf","ren_dead","ren_cf","dead_cf","compound"]):
        if key not in d:
            continue
        cond_vals = d[key]["vals"]
        n = min(len(clean_vals), len(cond_vals))
        if n >= 2:
            result[label] = [cond_vals[i] - clean_vals[i] for i in range(n)]
    return result

def from_curriculum(fpath):
    """Per-seed deltas from curriculum multiseed file (conditions.*.per_seed)."""
    d = json.load(open(fpath))
    conds_d = d["conditions"]
    seeds = [str(s) for s in d["seeds"]]
    clean_f1 = [conds_d["clean"]["per_seed"][s] for s in seeds]
    result = {}
    for label, key in zip(CONDS, ["ren","dead","cf","ren_dead","ren_cf","dead_cf","compound"]):
        if key not in conds_d:
            continue
        cond_f1 = [conds_d[key]["per_seed"][s] for s in seeds]
        result[label] = [cond_f1[i] - clean_f1[i] for i in range(len(seeds))]
    return result

# ── t-test helper ─────────────────────────────────────────────────────────────

def ttest(deltas):
    """One-sample t-test of deltas vs. 0. Returns (t, p, n)."""
    arr = np.array(deltas, dtype=float)
    if len(arr) < 2 or np.std(arr) == 0:
        return None, None, len(arr)
    t, p = stats.ttest_1samp(arr, 0)
    return float(t), float(p), len(arr)

# ── table definitions ─────────────────────────────────────────────────────────

BASE = "devign_full"

# ── DEVIGN (tab:devign) ───────────────────────────────────────────────────────
DEVIGN_MODELS = {
    "ECG RGCN":    lambda: from_7cond(f"{BASE}/ecgrgcn_7cond_results.json",     "test",      STD_COND_MAP),
    "ANGLE":       lambda: from_7cond(f"{BASE}/angle_7cond_results.json",        "test",      STD_COND_MAP),
    "VulGNN":      lambda: from_7cond(f"{BASE}/vulgnn_7cond_results.json",       "test",      STD_COND_MAP),
    "REVEAL":      lambda: from_7cond(f"{BASE}/reveal_7cond_results.json",       "originals", REVEAL_DEVIGN_MAP),
    "Dev.GGNN":    lambda: from_7cond(f"{BASE}/devign_ggnn_7cond_results.json",  "original",  REVEAL_DEVIGN_MAP),
    "LMGGNN":      lambda: from_7cond(f"{BASE}/lmggnn_7cond_results.json",       "test",      STD_COND_MAP),
    "CodeT5+":     lambda: from_7cond(f"{BASE}/codet5_7cond_devign_results.json","original",  REVEAL_DEVIGN_MAP),
    "TF-IDF+LR":   lambda: from_7cond(f"{BASE}/tfidf_7cond_devign_results.json", "test",      STD_COND_MAP),
    "CPG+LR":      lambda: from_7cond(f"{BASE}/cpglr_7cond_devign_results.json", "test",      STD_COND_MAP),
    "CodeBERT-Cur":lambda: from_curriculum(f"{BASE}/codebert_curriculum_multiseed_results.json"),
}

# ── BIGVUL (tab:bigvul) ───────────────────────────────────────────────────────
# Note: CodeT5+ and TF-IDF BigVul/DiverseVul use original/identifier keys (not test/test_obf_*)
BIGVUL_MODELS = {
    "ECG RGCN":  lambda: from_7cond(f"{BASE}/ecgrgcn_7cond_bigvul_results.json",   "test",     STD_COND_MAP),
    "ANGLE":     lambda: from_7cond(f"{BASE}/angle_7cond_bigvul_results.json",      "test",     STD_COND_MAP),
    "VulGNN":    lambda: from_7cond(f"{BASE}/vulgnn_7cond_bigvul_results.json",     "test",     STD_COND_MAP),
    "REVEAL":    lambda: from_7cond(f"{BASE}/reveal_7cond_bigvul_results.json",     "test",     STD_COND_MAP),
    "LMGGNN":    lambda: from_7cond(f"{BASE}/bigvul_lmggnn_7cond_results.json",     "test",     STD_COND_MAP),
    "CodeT5+":   lambda: from_7cond(f"{BASE}/codet5_7cond_bigvul_results.json",     "original", REVEAL_DEVIGN_MAP),
    "TF-IDF+LR": lambda: from_7cond(f"{BASE}/tfidf_7cond_bigvul_results.json",     "original", REVEAL_DEVIGN_MAP),
    "CPG+LR":    lambda: from_7cond(f"{BASE}/cpglr_7cond_bigvul_results.json",     "test",     STD_COND_MAP),
}

# ── DIVERSEVUL (tab:diversevul) ───────────────────────────────────────────────
DIVERSEVUL_MODELS = {
    "ECG RGCN":  lambda: from_7cond(f"{BASE}/ecgrgcn_7cond_diversevul_results.json",   "test",     STD_COND_MAP),
    "ANGLE":     lambda: from_7cond(f"{BASE}/angle_7cond_diversevul_results.json",      "test",     STD_COND_MAP),
    "VulGNN":    lambda: from_7cond(f"{BASE}/vulgnn_7cond_diversevul_results.json",     "test",     STD_COND_MAP),
    "REVEAL":    lambda: from_7cond(f"{BASE}/reveal_7cond_diversevul_results.json",     "test",     STD_COND_MAP),
    "LMGGNN":    lambda: from_7cond(f"{BASE}/diversevul_lmggnn_7cond_results.json",     "test",     STD_COND_MAP),
    "CodeT5+":   lambda: from_7cond(f"{BASE}/codet5_7cond_diversevul_results.json",     "original", REVEAL_DEVIGN_MAP),
    "TF-IDF+LR": lambda: from_7cond(f"{BASE}/tfidf_7cond_diversevul_results.json",     "original", REVEAL_DEVIGN_MAP),
    "CPG+LR":    lambda: from_7cond(f"{BASE}/cpglr_7cond_diversevul_results.json",     "test",     STD_COND_MAP),
}

# ── MITIGATION CodeBERT (tab:mitigation_codebert) ────────────────────────────
MITIG_CB_MODELS = {
    "CodeBERT":     lambda: from_7cond(f"{BASE}/codebert_7cond_devign_results.json", "original", REVEAL_DEVIGN_MAP),
    "CodeBERT-Aug": lambda: from_7cond(f"{BASE}/codebert_aug_7cond_results.json",    "test", STD_COND_MAP),
    "CodeBERT-Mix": lambda: from_multiseed_vals(f"{BASE}/codebert_mixed_multiseed_results.json"),
    "CodeBERT-Cur": lambda: from_curriculum(f"{BASE}/codebert_curriculum_multiseed_results.json"),
}

# ── MITIGATION ALL (tab:mitigation_all) ───────────────────────────────────────
MITIG_ALL_MODELS = {
    "REVEAL":      lambda: from_7cond(f"{BASE}/reveal_7cond_results.json",       "originals", REVEAL_DEVIGN_MAP),
    "REVEAL-Aug":  lambda: from_multiseed_vals(f"{BASE}/reveal_aug_multiseed_results.json"),
    "LMGGNN":      lambda: from_7cond(f"{BASE}/lmggnn_7cond_results.json",       "test",      STD_COND_MAP),
    "ReGVD":       lambda: from_7cond(f"{BASE}/regvd_7cond_devign_results.json", "test",      STD_COND_MAP),
    "ReGVD-Aug":   lambda: from_multiseed_vals(f"{BASE}/regvd_aug_multiseed_results.json"),
    "CodeBERT":    lambda: from_7cond(f"{BASE}/codebert_7cond_devign_results.json","original",  REVEAL_DEVIGN_MAP),
    "CodeBERT-Aug":lambda: from_7cond(f"{BASE}/codebert_aug_7cond_results.json",  "test",     STD_COND_MAP),
    "CodeBERT-Mix":lambda: from_multiseed_vals(f"{BASE}/codebert_mixed_multiseed_results.json"),
}

TABLES = {
    "Devign":       DEVIGN_MODELS,
    "BigVul":       BIGVUL_MODELS,
    "DiverseVul":   DIVERSEVUL_MODELS,
    "MitigCB":      MITIG_CB_MODELS,
    "MitigAll":     MITIG_ALL_MODELS,
}

# ── run tests and apply Bonferroni per table ──────────────────────────────────

all_results = {}

for table_name, model_dict in TABLES.items():
    table_results = {}

    # Pass 1: collect raw test results
    for model_name, loader in model_dict.items():
        fpath_check = None
        try:
            deltas_by_cond = loader()
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            continue

        model_tests = {}
        for cond, deltas in deltas_by_cond.items():
            t, p, n = ttest(deltas)
            model_tests[cond] = {
                "deltas": [round(x, 3) for x in deltas],
                "n": n,
                "t_stat": round(t, 4) if t is not None else None,
                "p_uncorrected": round(p, 6) if p is not None else None,
            }
        table_results[model_name] = model_tests

    # Pass 2: Bonferroni correction across all valid tests in this table
    all_p = []
    for model_name, model_tests in table_results.items():
        for cond, v in model_tests.items():
            if v["p_uncorrected"] is not None:
                all_p.append((model_name, cond, v["p_uncorrected"]))

    k = len(all_p)  # number of tests in this table
    for model_name, cond, p_raw in all_p:
        p_bonf = min(p_raw * k, 1.0)
        table_results[model_name][cond]["p_bonferroni"] = round(p_bonf, 6)
        table_results[model_name][cond]["n_bonferroni_tests"] = k
        table_results[model_name][cond]["sig_uncorrected"] = p_raw < 0.05
        table_results[model_name][cond]["sig_bonferroni"] = p_bonf < 0.05

    all_results[table_name] = table_results

# ── print summary ─────────────────────────────────────────────────────────────

for table_name, table_results in all_results.items():
    # count tests for this table
    k_tests = list(table_results.values())[0].get(CONDS[0], {}).get("n_bonferroni_tests", "?") if table_results else "?"
    print(f"\n{'='*80}")
    print(f"  TABLE: {table_name}  (k={k_tests} tests, Bonferroni α'=0.05/k)")
    print(f"  Markers: † = p<.05 uncorrected | * = p<.05 Bonferroni")
    print(f"{'='*80}")
    print(f"{'Model':<15}", end="")
    for c in CONDS:
        print(f"  {c:>8}", end="")
    print()
    print("-" * 80)

    for model_name, model_tests in table_results.items():
        print(f"{model_name:<15}", end="")
        for c in CONDS:
            v = model_tests.get(c)
            if v is None or v["p_uncorrected"] is None:
                print(f"  {'N/A':>8}", end="")
            else:
                mean_d = np.mean(v["deltas"])
                marker = ""
                if v.get("sig_bonferroni"):
                    marker = "*"
                elif v.get("sig_uncorrected"):
                    marker = "†"
                print(f"  {mean_d:+6.2f}{marker:1}", end="")
        print()

print(f"\n\nFull p-value report (sig_bonferroni only):")
print(f"{'Table':<12} {'Model':<15} {'Cond':<6} {'t':>7} {'p_raw':>9} {'p_bonf':>9} {'n':>3}")
print("-" * 65)
for table_name, table_results in all_results.items():
    for model_name, model_tests in table_results.items():
        for cond, v in model_tests.items():
            if v.get("sig_bonferroni"):
                print(f"{table_name:<12} {model_name:<15} {cond:<6} "
                      f"{v['t_stat']:+7.3f} {v['p_uncorrected']:9.5f} "
                      f"{v['p_bonferroni']:9.5f} {v['n']:>3}")

# ── save ─────────────────────────────────────────────────────────────────────

# Strip deltas from JSON to keep it compact
out = {}
for table_name, table_results in all_results.items():
    out[table_name] = {}
    for model_name, model_tests in table_results.items():
        out[table_name][model_name] = {}
        for cond, v in model_tests.items():
            entry = {k2: v2 for k2, v2 in v.items() if k2 != "deltas"}
            out[table_name][model_name][cond] = entry

with open("devign_full/significance_tests.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved → devign_full/significance_tests.json")
