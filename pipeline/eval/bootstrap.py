"""Test-sample uncertainty of Delta F1: paired bootstrap over test functions.

The intervals in results/stats/<ds>_valid.csv (pipeline.eval.stats) reflect training variability
(seeds or training subsamples). This script measures test-sample uncertainty instead:

  for each model x condition, the population is the validly transformed test functions used for the
  table entry (valid ids that have a clean and a transformed prediction; pipeline.eval.metrics);
  each replicate resamples those functions with replacement (one resample shared by every run, so
  clean and transformed stay paired and seeds are not flattened), recomputes F1 on clean and on
  transformed inputs for every run at its validation threshold, and averages the per-run deltas.
  The 95% interval is the 2.5/97.5 percentile over replicates.

  RERUN_ROOT=<root> python -m pipeline.eval.bootstrap --dataset devign [--reps 1000] [--out <csv>]
"""
import argparse
import csv
import gzip
import json

import numpy as np

from pipeline.eval.metrics import valid_ids
from pipeline.models.common import ROOT

MODELS = ["ecg_rgcn", "angle", "vulgnn", "reveal", "lmggnn", "regvd", "codebert", "codet5plus", "tfidf_lr", "cpg_lr"]
CONDS = ["identifier", "deadcode", "controlflow", "compound"]


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 200.0 * tp / np.maximum(d, 1), 0.0)


def one(dataset: str, model: str, cond: str, reps: int, seed: int) -> dict:
    runs = []
    for p in sorted((ROOT / "results" / "preds" / dataset / model).glob("run_*.json.gz")):
        with gzip.open(p, "rt") as f:
            r = json.load(f)
        runs.append((r["threshold"], r["test"]["clean"], r["test"][cond]))
    keep = None
    for _, cl, tr in runs:
        k = valid_ids(dataset, cond) & set(cl["ids"]) & set(tr["ids"])
        keep = k if keep is None else keep & k
    ids = sorted(keep)
    n = len(ids)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=reps).astype(np.float32)  # reps x n resample counts
    point, boot = [], []
    for thr, cl, tr in runs:
        yc = dict(zip(cl["ids"], cl["y"]))
        pc, pt = dict(zip(cl["ids"], cl["p"])), dict(zip(tr["ids"], tr["p"]))
        y = np.array([yc[i] for i in ids], dtype=np.float32)
        hc = np.array([pc[i] >= thr for i in ids], dtype=np.float32)
        ht = np.array([pt[i] >= thr for i in ids], dtype=np.float32)
        stats = []
        for h in (hc, ht):
            v = np.stack([y * h, (1 - y) * h, y * (1 - h)], axis=1)  # tp, fp, fn indicators
            stats.append((f1(*(v.sum(0))), f1(*(W @ v).T)))
        point.append(stats[1][0] - stats[0][0])
        boot.append(stats[1][1] - stats[0][1])
    boot = np.mean(boot, axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"dataset": dataset, "model": model, "condition": cond, "n": n, "n_runs": len(runs),
            "delta_f1": float(np.mean(point)), "ci_low": float(lo), "ci_high": float(hi), "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [one(a.dataset, m, c, a.reps, a.seed) for m in MODELS for c in CONDS
            if (ROOT / "results" / "preds" / a.dataset / m).exists()]
    out = a.out or str(ROOT / "results" / "stats" / f"{a.dataset}_bootstrap.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out, len(rows))


if __name__ == "__main__":
    main()
