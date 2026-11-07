"""The one significance procedure used for every robustness table.

Input: results/metrics/<dataset>/<model>.json (pipeline.eval.metrics). For each
model x condition, the per-run F1 deltas (transformed - clean, paired by run) are tested:

  neural detectors (independent training seeds):
      two-sided one-sample t-test of the per-seed deltas against 0 (= paired t-test),
      95% t-interval, Benjamini-Hochberg at q = 0.05 across all neural tests of the
      same paper table (dataset x scope x bh_family: main detectors / node-type
      ablation / training controls).
  non-neural detectors (TF-IDF+LR, CPG+LR; runs differ only by training subsample):
      95% percentile interval of the per-run deltas; not part of the BH family.

Scope "valid" (primary) uses only validly transformed samples, clean rescored on the
same ids; scope "all" uses every sample. Output: results/stats/<dataset>_<scope>.csv/json
with run counts, sample denominators, raw and BH-adjusted p-values.

  python -m pipeline.eval.stats --dataset devign
"""
import argparse
import csv
import json
import re

import numpy as np
from scipy import stats

from pipeline.models.common import CONDITIONS, ROOT

NON_NEURAL = {"tfidf_lr", "cpg_lr"}
Q = 0.05


def bh_family(model: str) -> str:
    """BH is applied within each paper table: main detectors, the node-type ablation, training controls."""
    if re.search(r"_(aug|volctl)", model):
        return "training"
    if model.endswith(("_struct", "_struct_cb")):
        return "ablation"
    return "main"


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj.tolist()


def table(dataset: str, scope: str) -> list[dict]:
    rows = []
    for f in sorted((ROOT / "results" / "metrics" / dataset).glob("*.json")):
        m = json.loads(f.read_text())
        for cond in CONDITIONS:
            r = m.get(scope, {}).get(cond)
            if not r:
                continue
            d = np.array(r["f1"]["delta_per_run"])
            row = {"dataset": dataset, "scope": scope, "model": m["model"], "condition": cond,
                   "family": "non-neural" if m["model"] in NON_NEURAL else "neural",
                   "bh_family": bh_family(m["model"]),
                   "n_runs": len(d), "n_samples": r["n"], "n_vul": r["n_vul"],
                   "clean_f1": r["f1"]["clean_mean"], "clean_f1_sd": r["f1"]["clean_std"],
                   "clean_f1_full_test": m["all"]["identifier"]["f1"]["clean_mean"],
                   "clean_f1_full_test_sd": m["all"]["identifier"]["f1"]["clean_std"],
                   "delta_f1": float(d.mean()),
                   "delta_precision": r["precision"]["delta_mean"], "delta_recall": r["recall"]["delta_mean"],
                   "delta_pr_auc": r["pr_auc"]["delta_mean"], "delta_roc_auc": r["roc_auc"]["delta_mean"],
                   "clean_ece": r["ece"]["clean_mean"], "delta_ece": r["ece"]["delta_mean"],
                   "clean_brier": r["brier"]["clean_mean"], "delta_brier": r["brier"]["delta_mean"]}
            if row["family"] == "non-neural":
                row["ci_low"], row["ci_high"] = np.percentile(d, [2.5, 97.5]).tolist()
                row["test"] = "percentile interval"
                row["significant"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)
            elif len(d) >= 2 and d.std(ddof=1) > 0:
                t, p = stats.ttest_1samp(d, 0.0)
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                row.update(test="paired t", t=float(t), p=float(p),
                           ci_low=float(d.mean() - half), ci_high=float(d.mean() + half))
            else:
                row.update(test="not testable (n<2 or zero variance)", p=None)
            rows.append(row)
    for name in ("main", "ablation", "training"):
        fam = [r for r in rows if r.get("p") is not None and r["bh_family"] == name]
        for r, a in zip(fam, bh([r["p"] for r in fam])):
            r["p_bh"] = a
            r["significant"] = bool(a < Q)
            r["bh_family_size"] = len(fam)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    for scope in ("valid", "all"):
        rows = table(args.dataset, scope)
        (out / f"{args.dataset}_{scope}.json").write_text(json.dumps(rows, indent=1))
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(out / f"{args.dataset}_{scope}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        sig = sum(bool(r.get("significant")) for r in rows)
        print(f"{args.dataset}/{scope}: {len(rows)} model x condition rows, {sig} significant")


if __name__ == "__main__":
    main()
