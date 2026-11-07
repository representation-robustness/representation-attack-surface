"""Does an independent rerun reproduce the findings of a reference run? Table-level comparison.

  python -m pipeline.report.compare_findings --ref REF_ROOT --new NEW_ROOT [--out FILE.md]

For every (dataset, model, condition) present in both results/stats/<ds>_valid.json:
  - clean F1 and Delta F1 in both runs, their difference;
  - whether the cell's significance verdict (bold) agrees;
  - sign agreement for cells where either |Delta| >= 0.5 pp;
  - whether the difference lies within run-to-run noise: |dDelta| <= 2 * sqrt(se_ref^2 + se_new^2), where se
    is the across-run standard error of each run's mean Delta (from the per-run deltas).
Plus the derived headline quantities: ablation Delta Ren, mitigation Max|Delta|, minimum-budget ASR@k,
correlation, and DeepWukong deltas. Writes a markdown report and prints a summary.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def stats(root, ds):
    p = root / "results" / "stats" / f"{ds}_valid.json"
    return {(r["model"], r["condition"]): r for r in json.loads(p.read_text())} if p.exists() else {}


def per_run_se(root, ds, model, cond):
    p = root / "results" / "metrics" / ds / f"{model}.json"
    if not p.exists():
        return None
    d = np.array(json.loads(p.read_text())["valid"].get(cond, {}).get("f1", {}).get("delta_per_run", []))
    return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None


def read_csv(root, name):
    p = root / "results" / "tables" / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    lines = [f"# Rerun vs reference\n\nreference: `{a.ref}`  \nrerun: `{a.new}`\n"]
    summary = []
    allrows = []
    for ds in ("devign", "diversevul", "bigvul"):
        r, n = stats(a.ref, ds), stats(a.new, ds)
        keys = sorted(r.keys() & n.keys())
        if not keys:
            continue
        rows = []
        for k in keys:
            x, y = r[k], n[k]
            se = [per_run_se(a.ref, ds, *k), per_run_se(a.new, ds, *k)]
            tol = 2 * np.sqrt(sum(s ** 2 for s in se)) if all(s is not None for s in se) else None
            dd = y["delta_f1"] - x["delta_f1"]
            rows.append({"ds": ds, "model": k[0], "cond": k[1], "ref": x["delta_f1"], "new": y["delta_f1"], "diff": dd,
                         "clean_ref": x.get("clean_f1_full_test", x["clean_f1"]), "clean_new": y.get("clean_f1_full_test", y["clean_f1"]),
                         "sig_ref": bool(x.get("significant")), "sig_new": bool(y.get("significant")),
                         "within_noise": None if tol is None else abs(dd) <= max(tol, 0.05),
                         "big": max(abs(x["delta_f1"]), abs(y["delta_f1"])) >= 0.5,
                         "sign_same": np.sign(x["delta_f1"]) == np.sign(y["delta_f1"])})
        allrows += rows
        big = [z for z in rows if z["big"]]
        noise = [z for z in rows if z["within_noise"] is not None]
        s = {"dataset": ds, "cells": len(rows),
             "sign_agree_big": f"{sum(z['sign_same'] for z in big)}/{len(big)}",
             "significance_agree": f"{sum(z['sig_ref'] == z['sig_new'] for z in rows)}/{len(rows)}",
             "within_noise": f"{sum(z['within_noise'] for z in noise)}/{len(noise)}",
             "mean_abs_diff_delta": round(float(np.mean([abs(z['diff']) for z in rows])), 3),
             "max_abs_diff_delta": round(float(max(abs(z['diff']) for z in rows)), 3),
             "mean_abs_diff_clean_f1": round(float(np.mean([abs(z['clean_new'] - z['clean_ref']) for z in rows if z['cond'] == 'identifier'])), 3)}
        summary.append(s)
        lines.append(f"\n## {ds}\n\n" + ", ".join(f"{k}: {v}" for k, v in s.items() if k != "dataset") + "\n")
        lines.append("| model | cond | ΔF1 ref | ΔF1 rerun | diff | sig ref/rerun | within noise |\n|---|---|---|---|---|---|---|")
        for z in sorted(rows, key=lambda z: -abs(z["diff"]))[:25]:
            lines.append(f"| {z['model']} | {z['cond']} | {z['ref']:+.2f} | {z['new']:+.2f} | {z['diff']:+.2f} | "
                         f"{'*' if z['sig_ref'] else '·'}/{'*' if z['sig_new'] else '·'} | {z['within_noise']} |")
    # derived tables
    for name, key, cols in [("table_ablation_{}.csv", ("base", "variant"), ["delta_ren", "diff_vs_base"]),
                            ("table_minbudget_{}.csv", ("model",), ["asr1", "asr3", "mean_budget"]),
                            ("table_mitigation_devign.csv", ("variant",), ["clean_f1", "max_abs_delta"])]:
        for ds in (["devign"] if "{}" not in name else ["devign", "diversevul", "bigvul"]):
            fn = name.format(ds)
            R = {tuple(x[k] for k in key): x for x in read_csv(a.ref, fn)}
            N = {tuple(x[k] for k in key): x for x in read_csv(a.new, fn)}
            if not R or not N:
                continue
            lines.append(f"\n## {fn}\n\n| row | " + " | ".join(f"{c} ref | {c} rerun" for c in cols) + " |\n|---|" + "---|" * (2 * len(cols)))
            for k in R.keys() & N.keys():
                cells = []
                for c in cols:
                    f = lambda v: f"{float(v):.2f}" if v not in ("", None) else "--"  # noqa: E731
                    cells += [f(R[k].get(c)), f(N[k].get(c))]
                lines.append(f"| {'/'.join(k)} | " + " | ".join(cells) + " |")
    for root, tag in ((a.ref, "ref"), (a.new, "rerun")):
        p = root / "results" / "tables" / "correlation.json"
        if p.exists():
            c = json.loads(p.read_text())
            lines.append(f"\ncorrelation ({tag}): " + ", ".join(f"{d}: r={v['pearson_r']:+.2f} (p={v['pearson_p']:.2f})" for d, v in c.items()))
    R, N = {x["condition"]: x for x in read_csv(a.ref, "table_deepwukong.csv")}, {x["condition"]: x for x in read_csv(a.new, "table_deepwukong.csv")}
    if R and N:
        lines.append("\n## DeepWukong\n\n| condition | ΔF1 ref | ΔF1 rerun | e2e ref | e2e rerun |\n|---|---|---|---|---|")
        for c in R.keys() & N.keys():
            lines.append(f"| {c} | {float(R[c]['delta_f1']):+.2f} | {float(N[c]['delta_f1']):+.2f} | "
                         f"{float(R[c]['e2e_delta_f1']):+.2f} | {float(N[c]['e2e_delta_f1']):+.2f} |")
    text = "\n".join(lines) + "\n"
    out = a.out or a.new / "results" / "compare_findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (out.with_suffix(".json")).write_text(json.dumps({"summary": summary, "cells": allrows}, indent=1, default=bool))
    for s in summary:
        print(s)
    print("report ->", out)


if __name__ == "__main__":
    main()
