"""
aggregate_results.py — Collect results from all baseline experiments and print
master summary tables (Devign, Big-Vul, DiverseVul) ready for Overleaf.

Usage:
    python utils/aggregate_results.py
"""

import json
import os

R = os.path.expanduser("~/thesis/devign_full")

ALL_OBF_KEYS = [
    "test_obf_identifier",
    "test_obf_deadcode",
    "test_obf_controlflow",
    "test_obf_ren_dead",
    "test_obf_ren_cf",
    "test_obf_dead_cf",
    "test_obf_compound",
]
SPLIT_KEYS   = ["test"] + ALL_OBF_KEYS
SPLIT_LABELS = ["Clean F1", "Ren", "Dead", "CF", "R+D", "R+CF", "D+CF", "Cmpd"]


def load_f1(path):
    """
    Load F1 values from a result JSON file.  Handles five key formats:
      A. Standard:   {"test": {"f1_mean": X}, "test_obf_identifier": {...}, ...}
      B. Short keys: {"original": {"f1_mean": X}, "identifier": {...}, ...}
      C. REVEAL-Dev: {"originals": {"f1_mean": X}, "identifier": {...}, ...}
      D. Nested:     {"results": {"original": {...}, ...}}
      E. DeepWukong: {"metrics": {"clean": {"f1": X}, "ren": {...}, ...}}
    Returns dict mapping canonical split key → f1 float, or None if file missing.
    """
    if not os.path.exists(path):
        return None

    with open(path) as f:
        d = json.load(f)

    def _extract(v):
        if isinstance(v, dict):
            return v.get("f1_mean", v.get("f1"))
        return float(v) if v is not None else None

    # Format E: DeepWukong — {"metrics": {"clean": {"f1": X}, "ren": {...}, ...}}
    if "metrics" in d and isinstance(d["metrics"], dict):
        m = d["metrics"]
        key_map = {
            "test":                 "clean",
            "test_obf_identifier":  "ren",
            "test_obf_deadcode":    "dead",
            "test_obf_controlflow": "cf",
            "test_obf_ren_dead":    "ren_dead",
            "test_obf_ren_cf":      "ren_cf",
            "test_obf_dead_cf":     "dead_cf",
            "test_obf_compound":    "compound",
        }
        return {can: _extract(m[loc]) for can, loc in key_map.items() if loc in m}

    # Format D: nested under "results"
    if "results" in d and isinstance(d["results"], dict):
        d = d["results"]

    # Format C: REVEAL Devign uses "originals" (with trailing s)
    if "originals" in d:
        key_map = {
            "test":                 "originals",
            "test_obf_identifier":  "identifier",
            "test_obf_deadcode":    "deadcode",
            "test_obf_controlflow": "controlflow",
            "test_obf_ren_dead":    "ren_dead",
            "test_obf_ren_cf":      "ren_cf",
            "test_obf_dead_cf":     "dead_cf",
            "test_obf_compound":    "compound",
        }
        return {can: _extract(d[loc]) for can, loc in key_map.items() if loc in d}

    # Format B: short keys ("original", "identifier", ...)
    if "original" in d and isinstance(d.get("original"), dict):
        key_map = {
            "test":                 "original",
            "test_obf_identifier":  "identifier",
            "test_obf_deadcode":    "deadcode",
            "test_obf_controlflow": "controlflow",
            "test_obf_ren_dead":    "ren_dead",
            "test_obf_ren_cf":      "ren_cf",
            "test_obf_dead_cf":     "dead_cf",
            "test_obf_compound":    "compound",
        }
        return {can: _extract(d[loc]) for can, loc in key_map.items() if loc in d}

    # Format A: standard "test" / "test_obf_*" keys
    return {k: _extract(d[k]) for k in SPLIT_KEYS if k in d and isinstance(d[k], dict)}


# ── Devign table ──────────────────────────────────────────────────────────────
# source = file path (str)  OR  hardcoded dict of {split_key: f1_value}
DEVIGN_SYSTEMS = [
    ("DeepWukong",      "Slice-based",   2021,
        os.path.join(R, "roc_data_dwk.json")),
    ("REVEAL",          "Whole-fn GNN",  2022,
        os.path.join(R, "reveal_7cond_results.json")),
    ("ECG RGCN",        "Whole-fn GNN",  2023,
        os.path.join(R, "ecgrgcn_7cond_results.json")),
    ("VulGNN",          "Whole-fn GNN",  2026,
        os.path.join(R, "vulgnn_7cond_results.json")),
    ("ANGLE",           "Whole-fn GNN",  2024,
        os.path.join(R, "angle_7cond_results.json")),
    ("Devign GGNN†","Whole-fn GNN", 2019,
        os.path.join(R, "devign_ggnn_7cond_results.json")),
    ("ReGVD",           "Graph + LM",    2022,
        os.path.join(R, "regvd_7cond_results.json")),
    ("Vul-LMGGNN",      "Graph + LM",    2023,
        os.path.join(R, "lmggnn_7cond_results.json")),
    ("CodeBERT",        "Transformer",   2020,
        os.path.join(R, "codebert_7cond_devign_results.json")),
    ("CodeBERT-Aug",    "Transformer",   2020,
        os.path.join(R, "codebert_aug_7cond_results.json")),
    ("CodeT5+",         "Transformer",   2023,
        os.path.join(R, "codet5_7cond_devign_results.json")),
    ("CPG + LogReg",    "Non-neural",    "ours",
        os.path.join(R, "cpglr_7cond_devign_results.json")),
    ("TF-IDF + LogReg", "Non-neural",    "ours",
        os.path.join(R, "tfidf_7cond_devign_results.json")),
]

# ── Big-Vul table ─────────────────────────────────────────────────────────────
BIGVUL_SYSTEMS = [
    ("REVEAL",          "Whole-fn GNN",  2022,
        os.path.join(R, "reveal_7cond_bigvul_results.json")),
    ("ECG RGCN",        "Whole-fn GNN",  2023,
        os.path.join(R, "ecgrgcn_7cond_bigvul_results.json")),
    ("VulGNN",          "Whole-fn GNN",  2026,
        os.path.join(R, "vulgnn_7cond_bigvul_results.json")),
    ("ANGLE",           "Whole-fn GNN",  2024,
        os.path.join(R, "angle_7cond_bigvul_results.json")),
    ("ReGVD",           "Graph + LM",    2022,
        os.path.join(R, "regvd_7cond_bigvul_results.json")),
    ("Vul-LMGGNN",      "Graph + LM",    2023,
        os.path.join(R, "bigvul_lmggnn_7cond_results.json")),
    ("CodeBERT",        "Transformer",   2020,
        os.path.join(R, "codebert_7cond_bigvul_results.json")),
    ("CodeT5+",         "Transformer",   2023,
        os.path.join(R, "codet5_7cond_bigvul_results.json")),
    ("CPG + LogReg",    "Non-neural",    "ours",
        os.path.join(R, "cpglr_7cond_bigvul_results.json")),
    ("TF-IDF + LogReg", "Non-neural",    "ours",
        os.path.join(R, "tfidf_7cond_bigvul_results.json")),
]

# ── DiverseVul table ──────────────────────────────────────────────────────────
DIVERSEVUL_SYSTEMS = [
    ("REVEAL",          "Whole-fn GNN",  2022,
        os.path.join(R, "reveal_7cond_diversevul_results.json")),
    ("ECG RGCN",        "Whole-fn GNN",  2023,
        os.path.join(R, "ecgrgcn_7cond_diversevul_results.json")),
    ("VulGNN",          "Whole-fn GNN",  2026,
        os.path.join(R, "vulgnn_7cond_diversevul_results.json")),
    ("ANGLE",           "Whole-fn GNN",  2024,
        os.path.join(R, "angle_7cond_diversevul_results.json")),
    ("ReGVD",           "Graph + LM",    2022,
        os.path.join(R, "regvd_7cond_diversevul_results.json")),
    ("Vul-LMGGNN",      "Graph + LM",    2023,
        os.path.join(R, "diversevul_lmggnn_7cond_results.json")),
    ("CodeBERT",        "Transformer",   2020,
        os.path.join(R, "codebert_7cond_diversevul_results.json")),
    ("CodeT5+",         "Transformer",   2023,
        os.path.join(R, "codet5_7cond_diversevul_results.json")),
    ("CPG + LogReg",    "Non-neural",    "ours",
        os.path.join(R, "cpglr_7cond_diversevul_results.json")),
    ("TF-IDF + LogReg", "Non-neural",    "ours",
        os.path.join(R, "tfidf_7cond_diversevul_results.json")),
]


def get_f1s(source):
    if isinstance(source, dict):
        return {k: source.get(k) for k in SPLIT_KEYS}
    return load_f1(source) or {k: None for k in SPLIT_KEYS}


def delta_str(clean, obf):
    if clean is None or obf is None:
        return "pend."
    d = obf - clean
    sign = "+" if d >= 0 else ""
    return f"{obf:.1f}({sign}{d:.1f})"


def print_table(systems, title):
    col_w = 13
    header_parts = [f"{'System':<20}", f"{'Year':<6}", f"{'Clean':>7}"]
    header_parts += [f"{lbl:>{col_w}}" for lbl in SPLIT_LABELS[1:]]
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    print("  " + "  ".join(header_parts))
    print(f"  {'-' * 126}")

    for (name, family, year, source) in systems:
        f1s   = get_f1s(source)
        clean = f1s.get("test")
        row   = [f"{name:<20}", f"{str(year):<6}",
                 f"{(f'{clean:.1f}' if clean is not None else 'pend.'):>7}"]
        row  += [f"{delta_str(clean, f1s.get(k)):>{col_w}}" for k in ALL_OBF_KEYS]
        print("  " + "  ".join(row))

    print(f"  {'=' * 126}")
    print(f"  Columns: Clean F1 (%) | Obf columns show F1(ΔF1) — positive Δ means improvement")


def latex_row(name, year, f1s):
    clean = f1s.get("test")
    if clean is None:
        return f"% {name}: missing"
    obfs = [f1s.get(k) for k in ALL_OBF_KEYS]

    def delta_cmd(ref, val):
        if val is None:
            return "\\pend{}"
        d = round(val - ref, 2)
        return f"\\up{{{abs(d):.2f}}}" if d >= 0 else f"\\dn{{{abs(d):.2f}}}"

    def f1_str(v):
        return f"{v:.2f}" if v is not None else "---"

    cols = " & ".join(
        f"{f1_str(v)} & {delta_cmd(clean, v)}" for v in obfs
    )
    return f"{name} & {year} & {clean:.2f} & {cols} \\\\"


def print_latex(systems, label):
    print(f"\n% ── LaTeX rows: {label} ──")
    for (name, family, year, source) in systems:
        f1s = get_f1s(source)
        print(latex_row(name, year, f1s))


def main():
    print_table(DEVIGN_SYSTEMS,    "DEVIGN — Trained on Devign, evaluated on all 7 obfuscation conditions")
    print_table(BIGVUL_SYSTEMS,    "BIG-VUL — Trained on Big-Vul, evaluated on all 7 obfuscation conditions")
    print_table(DIVERSEVUL_SYSTEMS,"DIVERSEVUL — Trained on DiverseVul, evaluated on all 7 obfuscation conditions")

    print("\n\n" + "=" * 80)
    print("LaTeX rows (\\up{X} = +X pp, \\dn{X} = −X pp)")
    print("=" * 80)
    print_latex(DEVIGN_SYSTEMS,    "Devign")
    print_latex(BIGVUL_SYSTEMS,    "Big-Vul")
    print_latex(DIVERSEVUL_SYSTEMS,"DiverseVul")


if __name__ == "__main__":
    main()
