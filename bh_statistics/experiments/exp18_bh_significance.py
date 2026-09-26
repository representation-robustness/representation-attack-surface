"""Paired significance tests with Benjamini-Hochberg (BH) correction, recomputed from the per-run values stored in devign_full/*.

Usage (from the repository root):  python experiments/exp18_bh_significance.py
Outputs: devign_full/bh_paired_tests_neural.csv, bh_nonneural_intervals.csv, bh_summary.json, bh_grids.txt

Design (stated so it can be reviewed):
  * Test: two-sided one-sample t-test on per-run deltas (condition F1 minus clean F1, paired by run index),
          H0: mean delta = 0. Equivalent to a paired t-test. Needs >=2 runs and non-zero variance.
  * Family for multiple-comparison correction: all valid neural tests within one table (dataset).
    BH: standard step-up procedure, q < 0.05. Bonferroni: p * m, capped at 1.
  * Non-neural baselines (TF-IDF, CPG+LR) use bootstrap trials, not independent seeds, so they are reported with a
    95% percentile interval of per-trial deltas and are NOT included in the BH/Bonferroni family (as in the paper).
  * Excluded and flagged: rows with no per-run values (ReGVD), single-checkpoint rows (Devign CodeBERT, n=1),
    REVEAL Devign R+D / R+CF / D+CF (stored per-run values are the failed evaluation, F1 about 1%),
    Devign GGNN R+D / R+CF (copied from Dead / CF in the source file, not measured).
"""
import argparse, json, csv, os, datetime
from pathlib import Path
import numpy as np
from scipy import stats

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="repository root (default: parent of experiments/)")
_ap.add_argument("--out", default=None, help="output directory (default: <repo>/devign_full)")
_args = _ap.parse_args()
D = os.path.join(_args.repo, "devign_full") + os.sep
OUT = _args.out or D
os.makedirs(OUT, exist_ok=True)

CLEAN = ("test", "original", "originals")
COND_KEYS = {
    "Ren": ("test_obf_identifier", "identifier"), "Dead": ("test_obf_deadcode", "deadcode"),
    "CF": ("test_obf_controlflow", "controlflow"), "R+D": ("test_obf_ren_dead", "ren_dead"),
    "R+CF": ("test_obf_ren_cf", "ren_cf"), "D+CF": ("test_obf_dead_cf", "dead_cf"),
    "Cmp": ("test_obf_compound", "compound"),
}
CONDS = list(COND_KEYS)


def runs(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            if "all_f1" in v: return v["all_f1"]
            if "vals" in v: return v["vals"]
    return None


def load(fname):
    d = json.load(open(D + fname))
    return d["results"] if isinstance(d.get("results"), dict) else d


# (table, model, file, family) ; family: neural | nonneural
ROWS = []
DEV = [("ECG RGCN", "ecgrgcn_7cond_results.json"), ("ANGLE", "angle_7cond_results.json"), ("VulGNN", "vulgnn_7cond_results.json"),
       ("REVEAL", "reveal_7cond_results.json"), ("Dev.GGNN", "devign_ggnn_7cond_results.json"), ("Vul-LMGGNN", "lmggnn_7cond_results.json"),
       ("ReGVD", "regvd_7cond_devign_results.json"), ("CodeBERT", "codebert_7cond_devign_results.json"), ("CodeT5+", "codet5_7cond_devign_results.json"),
       ("CodeBERT-Aug", "codebert_aug_7cond_results.json")]
DIV = [("ECG RGCN", "ecgrgcn_7cond_diversevul_results.json"), ("ANGLE", "angle_7cond_diversevul_results.json"), ("VulGNN", "vulgnn_7cond_diversevul_results.json"),
       ("REVEAL", "reveal_7cond_diversevul_results.json"), ("Vul-LMGGNN", "diversevul_lmggnn_7cond_results.json"), ("ReGVD", "regvd_7cond_diversevul_results.json"),
       ("CodeBERT", "codebert_7cond_diversevul_results.json"), ("CodeT5+", "codet5_7cond_diversevul_results.json")]
BIG = [("ECG RGCN", "ecgrgcn_7cond_bigvul_results.json"), ("ANGLE", "angle_7cond_bigvul_results.json"), ("VulGNN", "vulgnn_7cond_bigvul_results.json"),
       ("REVEAL", "reveal_7cond_bigvul_results.json"), ("Vul-LMGGNN", "bigvul_lmggnn_7cond_results.json"), ("ReGVD", "regvd_7cond_bigvul_results.json"),
       ("CodeBERT", "codebert_7cond_bigvul_results.json"), ("CodeT5+", "codet5_7cond_bigvul_results.json")]
NONNEURAL = {"Devign (III)": [("TF-IDF+LR", "tfidf_7cond_devign_results.json"), ("CPG+LR", "cpglr_7cond_devign_results.json")],
             "DiverseVul (IV)": [("TF-IDF+LR", "tfidf_7cond_diversevul_results.json"), ("CPG+LR", "cpglr_7cond_diversevul_results.json")],
             "Big-Vul (V)": [("TF-IDF+LR", "tfidf_7cond_bigvul_results.json"), ("CPG+LR", "cpglr_7cond_bigvul_results.json")]}
TABLES = {"Devign (III)": DEV, "DiverseVul (IV)": DIV, "Big-Vul (V)": BIG}

EXCLUDE = {("Devign (III)", "REVEAL", c): "stored per-run values are the failed evaluation (F1 about 1%)" for c in ("R+D", "R+CF", "D+CF")}
EXCLUDE.update({("Devign (III)", "Dev.GGNN", c): "entry copied from Dead/CF in source file, not measured" for c in ("R+D", "R+CF")})

records = []   # one per (table, model, cond)


def add_neural(table, model, clean, cond_runs, source, note_default=""):
    for c in CONDS:
        r = cond_runs.get(c)
        rec = dict(table=table, model=model, cond=c, source=source, family="neural", n=None, mean_delta=None, sd=None, t=None,
                   p=None, note=note_default)
        if (table, model, c) in EXCLUDE:
            rec["note"] = "EXCLUDED: " + EXCLUDE[(table, model, c)]
        elif clean is None or r is None:
            rec["note"] = "EXCLUDED: no per-run values"
        elif len(clean) != len(r):
            rec["note"] = f"EXCLUDED: run counts differ ({len(clean)} vs {len(r)})"
        elif len(clean) < 2:
            rec["note"] = f"EXCLUDED: only {len(clean)} run (single checkpoint)"
        else:
            delta = np.array(r, float) - np.array(clean, float)
            rec.update(n=len(delta), mean_delta=float(delta.mean()), sd=float(delta.std(ddof=1)))
            if np.allclose(delta, delta[0], atol=1e-12) and rec["sd"] == 0.0:
                rec["note"] = "not testable: identical deltas across runs (zero variance)"
            else:
                t, p = stats.ttest_1samp(delta, 0.0)
                rec.update(t=float(t), p=float(p))
        records.append(rec)


for table, rows in TABLES.items():
    for model, f in rows:
        d = load(f)
        clean = runs(d, CLEAN)
        cr = {c: runs(d, COND_KEYS[c]) for c in CONDS}
        add_neural(table, model, clean, cr, f)

# Table VI ablation arms (Ren only) and Table IX augmented models (7 conds), from multiseed files
def multiseed(fname):
    d = load(fname)
    cl = d["clean"]["vals"]
    m = {"Ren": "ren", "Dead": "dead", "CF": "cf", "R+D": "ren_dead", "R+CF": "ren_cf", "D+CF": "dead_cf", "Cmp": "compound"}
    return cl, {c: d[k]["vals"] for c, k in m.items()}

for model, f, tab in [("REVEAL-StructOnly", "reveal_noid_multiseed_results.json", "VI"), ("VulGNN+CodeBERT", "vulgnn_withid_multiseed_results.json", "VI")]:
    cl, cr = multiseed(f)
    start = len(records); add_neural("Table VI (ablation)", model, cl, {"Ren": cr["Ren"]}, f)
    # only Ren is of interest here: drop the other 6 placeholder rows added by add_neural
    records[:] = [r for i, r in enumerate(records) if not (i >= start and r["cond"] != "Ren")]
for model, f in [("CodeBERT-Aug", "codebert_aug_7cond_results.json"), ("ReGVD-Aug", "regvd_aug_multiseed_results.json"), ("REVEAL-Aug", "reveal_aug_multiseed_results.json")]:
    if model == "CodeBERT-Aug":
        continue  # already in Devign table
    cl, cr = multiseed(f)
    add_neural("Table IX (augmented)", model, cl, cr, f)
cl_cb = runs(load("codebert_aug_7cond_results.json"), CLEAN)
add_neural("Table IX (augmented)", "CodeBERT-Aug", cl_cb, {c: runs(load("codebert_aug_7cond_results.json"), COND_KEYS[c]) for c in CONDS}, "codebert_aug_7cond_results.json")

# multiple-comparison correction within each neural family (table)
fam = {}
for r in records:
    if r["p"] is not None: fam.setdefault(r["table"], []).append(r)
for table, rs in fam.items():
    m = len(rs); ps = np.array([r["p"] for r in rs]); order = np.argsort(ps)
    q = np.empty(m); prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]; prev = min(prev, ps[i] * m / rank); q[i] = prev
    for r, qi in zip(rs, q):
        r["m_tests"] = m; r["p_bonf"] = min(1.0, r["p"] * m); r["q_bh"] = float(qi)
        r["sig_uncorrected"] = r["p"] < 0.05; r["sig_bonf"] = r["p_bonf"] < 0.05; r["sig_bh"] = r["q_bh"] < 0.05

# non-neural: interval of per-trial deltas (not part of BH family)
nonneural = []
for table, rows in NONNEURAL.items():
    for model, f in rows:
        d = load(f); clean = runs(d, CLEAN)
        for c in CONDS:
            r = runs(d, COND_KEYS[c])
            if clean is None or r is None or len(clean) != len(r):
                continue
            delta = np.array(r, float) - np.array(clean, float)
            lo, hi = np.percentile(delta, [2.5, 97.5])
            nonneural.append(dict(table=table, model=model, cond=c, n=len(delta), mean_delta=float(delta.mean()), ci_lo=float(lo), ci_hi=float(hi),
                                  excludes_zero=bool(lo > 0 or hi < 0), source=f,
                                  note="trials" if len(delta) >= 30 else f"only {len(delta)} runs; interval not meaningful"))

# sanity check against the artifact's existing significance_tests.json (uncorrected p)
sig = json.load(open(D + "significance_tests.json"))
name_map = {"ECG RGCN": "ECG RGCN", "ANGLE": "ANGLE", "VulGNN": "VulGNN", "REVEAL": "REVEAL", "Dev.GGNN": "Dev.GGNN", "Vul-LMGGNN": "LMGGNN", "CodeT5+": "CodeT5+"}
tab_map = {"Devign (III)": "Devign", "DiverseVul (IV)": "DiverseVul", "Big-Vul (V)": "BigVul"}
diffs = []
for r in records:
    if r["p"] is None or r["table"] not in tab_map or r["model"] not in name_map: continue
    old = sig.get(tab_map[r["table"]], {}).get(name_map[r["model"]], {}).get(r["cond"])
    if isinstance(old, dict) and old.get("p_uncorrected") is not None:
        diffs.append(abs(old["p_uncorrected"] - r["p"]))
check = dict(n_compared=len(diffs), max_abs_diff=max(diffs) if diffs else None)

cols = ["table", "model", "cond", "n", "mean_delta", "sd", "t", "p", "m_tests", "p_bonf", "q_bh", "sig_uncorrected", "sig_bonf", "sig_bh", "note", "source"]
with open(os.path.join(OUT, "bh_paired_tests_neural.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in records: w.writerow({k: ("" if r.get(k) is None else (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k))) for k in cols})
with open(os.path.join(OUT, "bh_nonneural_intervals.csv"), "w", newline="") as fh:
    ncols = ["table", "model", "cond", "n", "mean_delta", "ci_lo", "ci_hi", "excludes_zero", "note", "source"]
    w = csv.DictWriter(fh, fieldnames=ncols); w.writeheader()
    for r in nonneural: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ncols})
summary = {}
for table, rs in fam.items():
    summary[table] = dict(m_tests=len(rs), sig_uncorrected=sum(r["sig_uncorrected"] for r in rs), sig_bonferroni=sum(r["sig_bonf"] for r in rs),
                          sig_bh=sum(r["sig_bh"] for r in rs),
                          bh_only=[f'{r["model"]}/{r["cond"]}' for r in rs if r["sig_bh"] and not r["sig_bonf"]],
                          bonf_not_bh=[f'{r["model"]}/{r["cond"]}' for r in rs if r["sig_bonf"] and not r["sig_bh"]])
excluded = [dict(table=r["table"], model=r["model"], cond=r["cond"], reason=r["note"]) for r in records if r["note"].startswith("EXCLUDED") or r["note"].startswith("not testable")]
json.dump(dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), scipy=__import__("scipy").__version__, summary=summary,
               sanity_check_vs_significance_tests_json=check, excluded_or_untestable=excluded),
          open(os.path.join(OUT, "bh_summary.json"), "w"), indent=1)
order = CONDS
grid = ["BH significance grid (B = significant at BH q<0.05, u = significant only before correction, . = not significant, - = not testable (zero variance), x = excluded (no valid per-run data))\n"]
for tname in ("Devign (III)", "DiverseVul (IV)", "Big-Vul (V)"):
    trs = [r for r in records if r["table"] == tname]; models = []
    for r in trs:
        if r["model"] not in models: models.append(r["model"])
    grid.append(f"\n{tname}\n{'model':13s} " + " ".join(f"{c:>5s}" for c in order))
    for m in models:
        cells = []
        for c in order:
            r = [x for x in trs if x["model"] == m and x["cond"] == c][0]
            g = "x" if r["note"].startswith("EXCLUDED") else "-" if r["p"] is None else "B" if r.get("sig_bh") else "u" if r.get("sig_uncorrected") else "."
            cells.append(f"{g:>5s}")
        grid.append(f"{m:13s} " + " ".join(cells))
open(os.path.join(OUT, "bh_grids.txt"), "w").write("\n".join(grid) + "\n")
print(json.dumps(dict(summary=summary, check=check), indent=1))
print("excluded / untestable cells:", len(excluded))
