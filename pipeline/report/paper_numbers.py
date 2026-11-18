"""In-text numbers of the paper that are not cells of a generated table.

Every value is computed from saved predictions, validity records, attack results or clean splits under
$RERUN_ROOT; nothing is typed by hand. Writes results/tables/paper_numbers.json (machine-readable) and
results/tables/paper_numbers.md (what each value is and where the paper uses it).

  python -m pipeline.report.paper_numbers
"""
import csv
import glob
import gzip
import json
import re
from collections import defaultdict

import numpy as np

from pipeline.eval.stats import bh
from pipeline.models.common import ROOT

DS = ["devign", "diversevul", "bigvul", "primevul"]
MAIN = ["ecg_rgcn", "angle", "vulgnn", "reveal", "lmggnn", "regvd", "codebert", "codet5plus", "tfidf_lr", "cpg_lr"]
STRUCT = ["ecg_rgcn_struct", "angle_struct", "vulgnn_struct", "reveal_struct"]
SINGLE = ["identifier", "deadcode", "controlflow"]
TRANSFER7 = ["angle", "vulgnn", "ecg_rgcn", "reveal", "regvd", "codebert", "codet5plus"]
R = ROOT / "results"


def jl(p):
    return [json.loads(l) for l in open(p)]


def runs(ds, m):
    for f in sorted(glob.glob(str(R / "preds" / ds / m / "run_*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            yield json.load(fh)


def f1(y, h):
    y, h = np.asarray(y, bool), np.asarray(h, bool)
    tp, fp, fn = (y & h).sum(), (~y & h).sum(), (y & ~h).sum()
    return 200 * tp / max(1, 2 * tp + fp + fn)


def rng(v, nd=1):
    return [round(float(min(v)), nd), round(float(max(v)), nd)]


def stat_rows(ds):
    return list(csv.DictReader(open(R / "stats" / f"{ds}_valid.csv")))


def transforms():
    """Literal-0 predicate share (valid dead-code functions) and fallback-only share (valid control-flow functions)."""
    out = {}
    for ds in DS:
        v = ROOT / "data" / ds / "validity" / "test"
        dead = [r for r in jl(v / "deadcode.jsonl") if r["valid"]]
        cf = [r for r in jl(v / "controlflow.jsonl") if r["valid"]]
        lit = sum(next(t for t in r["transforms"] if t["transform"] == "deadcode").get("opaque_var") is None for r in dead)
        fb = sum(bool(next(t for t in r["transforms"] if t["transform"] == "controlflow").get("fallback")) for r in cf)
        out[ds] = {"literal_predicate_pct": 100 * lit / len(dead), "cf_fallback_only_pct": 100 * fb / len(cf)}
    return out


def setup():
    """Graph coverage of the test split; mean training-function length by label; paired-vs-full clean F1 gap."""
    out = {}
    for ds in DS:
        g = len(next(runs(ds, "ecg_rgcn"))["test"]["clean"]["ids"])
        t = len(next(runs(ds, "codebert"))["test"]["clean"]["ids"])
        lens = defaultdict(list)
        for r in jl(ROOT / "data" / ds / "clean" / "train.jsonl"):
            lens[r["target"]].append(len(r["func"].split("\n")))
        gap = max(abs(float(r["clean_f1"]) - float(r["clean_f1_full_test"])) for r in stat_rows(ds) if r["model"] in MAIN)
        out[ds] = {"graph_coverage_pct": 100 * g / t, "train_lines_vul": float(np.mean(lens[1])),
                   "train_lines_nonvul": float(np.mean(lens[0])), "max_paired_vs_full_clean_f1_gap_pp": gap}
    return out


def leakage():
    """Whitespace-insensitive exact duplicates of test functions in the training split; Big-Vul clean F1 without them."""
    key = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    out = {}
    for ds in DS:
        train = {key(r["func"]) for r in jl(ROOT / "data" / ds / "clean" / "train.jsonl")}
        test = jl(ROOT / "data" / ds / "clean" / "test.jsonl")
        dup = {r["id"] for r in test if key(r["func"]) in train}
        vul = [r for r in test if r["target"] == 1]
        out[ds] = {"test_dup_pct": 100 * len(dup) / len(test),
                   "vul_test_dup_pct": 100 * sum(r["id"] in dup for r in vul) / max(1, len(vul))}
        if ds == "bigvul":
            change = {}
            for m in MAIN:
                d = []
                for run in runs(ds, m):
                    c, t = run["test"]["clean"], run["threshold"]
                    h = [p >= t for p in c["p"]]
                    keep = [i for i, x in enumerate(c["ids"]) if x not in dup]
                    d.append(f1([c["y"][i] for i in keep], [h[i] for i in keep]) - f1(c["y"], h))
                change[m] = float(np.mean(d))
            out[ds]["dedup_clean_f1_change_pp"] = change
            out[ds]["dedup_max_abs_change_pp"] = max(abs(v) for v in change.values())
    return out


def fixed_policy():
    """Operational fixed-policy rate (evaded / |X_f|, mean over runs), coverage on X_f, and the clean-correct share
    of the vulnerable functions each detector scores (= clean recall)."""
    out = {}
    for ds in DS:
        per = {}
        for m in MAIN:
            a = json.loads((R / "attacks" / ds / f"minbudget_{m}.json").read_text())
            pr = a["per_run"]
            op = {c: float(np.mean([100 * r["fixed_policy"][c]["evaded"] / max(1, r["denominator"]) for r in pr])) for c in SINGLE}
            cov = {c: float(np.mean([100 * r["fixed_policy"][c]["denominator"] / max(1, r["denominator"]) for r in pr])) for c in SINGLE}
            rec = [100 * sum(1 for y, p in zip(run["test"]["clean"]["y"], run["test"]["clean"]["p"]) if y == 1 and p >= run["threshold"])
                   / max(1, sum(run["test"]["clean"]["y"])) for run in runs(ds, m)]
            per[m] = {"operational": op, "coverage_on_Xf": cov, "clean_correct_share_of_vul": float(np.mean(rec))}
        best = max(((v["operational"][c], m, c) for m, v in per.items() for c in SINGLE))
        out[ds] = {"per_model": per, "max_operational": {"value": best[0], "model": best[1], "condition": best[2]},
                   "coverage_range": rng([v["coverage_on_Xf"][c] for v in per.values() for c in SINGLE]),
                   "clean_correct_share_range": rng([v["clean_correct_share_of_vul"] for v in per.values()])}
    return out


def struct_variants():
    """Node-type-only variants: clean F1 vs. an all-positive predictor on the same (graph-scored) test set,
    and the share of predictions that renaming changes."""
    out = {}
    for ds in DS:
        y = next(runs(ds, "ecg_rgcn_struct"))["test"]["clean"]["y"]
        p = sum(y) / len(y)
        rows = {r["model"]: r for r in stat_rows(ds) if r["condition"] == "identifier"}
        valid = {r["id"] for r in jl(ROOT / "data" / ds / "validity" / "test" / "identifier.jsonl") if r["valid"]}
        flips = {}
        for m in STRUCT:
            fl = n = 0
            for run in runs(ds, m):
                t = run["threshold"]
                c = dict(zip(run["test"]["clean"]["ids"], run["test"]["clean"]["p"]))
                q = dict(zip(run["test"]["identifier"]["ids"], run["test"]["identifier"]["p"]))
                ids = [i for i in valid if i in c and i in q]
                n += len(ids)
                fl += sum((c[i] >= t) != (q[i] >= t) for i in ids)
            flips[m] = 100 * fl / max(1, n)
        out[ds] = {"all_positive_f1": 200 * p / (1 + p),
                   "struct_clean_f1_range": rng([float(rows[m]["clean_f1_full_test"]) for m in STRUCT]),
                   "renaming_prediction_change_pct": flips}
    out["max_renaming_prediction_change_pct"] = max(v for ds in DS for v in out[ds]["renaming_prediction_change_pct"].values())
    return out


def ablation_bh():
    """BH-adjusted q over the 16 Welch tests of the change in Delta-Ren (four GNNs x four datasets)."""
    p = []
    for ds in DS:
        for r in csv.DictReader(open(R / "tables" / f"table_ablation_{ds}.csv")):
            if r["variant"].endswith("_struct") and r["variant"] != r["base"] and r["diff_p_welch"]:
                p.append(float(r["diff_p_welch"]))
    q = bh(p)
    return {"n_tests": len(p), "max_q": max(q)}


def transfer(fp):
    """Devign transfer: yield over the jointly correct population, TASR std, and how many pairs exceed the
    target's best single-family operational fixed-policy rate."""
    rows = [r for r in csv.DictReader(open(R / "tables" / "table_transfer_devign.csv"))
            if r["source"] in TRANSFER7 and r["target"] in TRANSFER7]
    best = {m: max(fp["devign"]["per_model"][m]["operational"].values()) for m in TRANSFER7}
    return {"n_pairs": len(rows),
            "yield_pct_range": rng([100 * float(r["transferred_mean"]) / float(r["joint_correct_mean"]) for r in rows], 2),
            "tasr_sd_range": rng([float(r["tasr_sd"]) for r in rows]),
            "pairs_above_target_best_fixed": sum(float(r["tasr_mean"]) > best[r["target"]] for r in rows)}


def interval_widths():
    """Median width of the seed (t/percentile) and paired-bootstrap 95% intervals over the 40 headline entries."""
    out = {}
    for ds in DS:
        seed = {(r["model"], r["condition"]): float(r["ci_high"]) - float(r["ci_low"]) for r in stat_rows(ds) if r["ci_high"]}
        bs = list(csv.DictReader(open(R / "stats" / f"{ds}_bootstrap.csv")))
        out[ds] = {"bootstrap_median_width": float(np.median([float(r["ci_high"]) - float(r["ci_low"]) for r in bs])),
                   "seed_median_width": float(np.median([seed[(r["model"], r["condition"])] for r in bs])),
                   "bootstrap_narrower_count": sum(float(r["ci_high"]) - float(r["ci_low"]) < seed[(r["model"], r["condition"])] for r in bs),
                   "n": len(bs)}
    return out


def deepwukong_sd():
    """Per-seed std of transformed F1 on the paired XFGs (Table X), same scope as pipeline.deepwukong.report."""
    from pipeline.deepwukong.report import _m, load_index
    from pipeline.deepwukong.common import CONDITIONS as DC
    rr = [json.load(gzip.open(f)) for f in sorted(glob.glob(str(R / "preds/deepwukong/deepwukong/run_*.json.gz")))]
    clean_idx = load_index("clean")
    out = {}
    for c in DC[1:]:
        idx = load_index(c)
        scope = {i for i, x in idx.items() if x["found"] and x["valid_scope"] and clean_idx.get(i, {}).get("found")}
        vals = []
        for r in rr:
            cl, tr = r["test"]["clean"], r["test"][c]
            pc, pt, yc = dict(zip(cl["ids"], cl["p"])), dict(zip(tr["ids"], tr["p"])), dict(zip(cl["ids"], cl["y"]))
            ids = sorted(scope & pc.keys() & pt.keys())
            vals.append(_m([yc[i] for i in ids], [pt[i] for i in ids], r["threshold"])["f1"])
        out[c] = {"f1_mean": float(np.mean(vals)), "f1_sd": float(np.std(vals, ddof=1))}
    return out


def main():
    fp = fixed_policy()
    res = {"transforms": transforms(), "setup": setup(), "leakage": leakage(), "fixed_policy": fp,
           "struct_variants": struct_variants(), "ablation_bh": ablation_bh(), "transfer_devign": transfer(fp),
           "interval_widths": interval_widths(), "deepwukong_paired_f1_sd": deepwukong_sd() if glob.glob(str(R / "preds/deepwukong/deepwukong/run_*.json.gz")) else None}
    (R / "tables" / "paper_numbers.json").write_text(json.dumps(res, indent=1))
    md = ["# In-text numbers (generated by pipeline/report/paper_numbers.py)", ""]
    for k, v in res.items():
        md += [f"## {k}", "", "```", json.dumps(v, indent=1), "```", ""]
    (R / "tables" / "paper_numbers.md").write_text("\n".join(md))
    print(R / "tables" / "paper_numbers.json")


if __name__ == "__main__":
    main()
