"""Metrics from per-run prediction files.

For each run and test condition we report, on two sample sets:
  all   : every test sample the model produced a prediction for
  valid : samples whose transformation for that condition passed the validity
          gate; the clean score is recomputed on exactly the same ids, so the
          delta compares the same functions before and after transformation.
Metrics: F1/precision/recall (validation-chosen threshold), PR-AUC, ROC-AUC, calibration
(Brier score, 10-bin expected calibration error of p), confusion counts, n, n_vul.

  python -m pipeline.eval.metrics --dataset devign --model tfidf_lr
"""
import argparse
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from pipeline.models.common import CONDITIONS, ROOT


@lru_cache(maxsize=None)
def valid_ids(dataset: str, condition: str) -> frozenset:
    p = ROOT / "data" / dataset / "validity" / "test" / f"{condition}.jsonl"
    return frozenset(r["id"] for r in map(json.loads, open(p)) if r["valid"])


def scores(y, p, thr) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= thr).astype(int)
    two = len(set(y.tolist())) == 2
    return {"n": int(len(y)), "n_vul": int(y.sum()),
            "f1": 100 * f1_score(y, yhat, zero_division=0),
            "precision": 100 * precision_score(y, yhat, zero_division=0),
            "recall": 100 * recall_score(y, yhat, zero_division=0),
            "pr_auc": 100 * average_precision_score(y, p) if two else float("nan"),
            "roc_auc": 100 * roc_auc_score(y, p) if two else float("nan"),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
            "tp": int(((yhat == 1) & (y == 1)).sum()), "fp": int(((yhat == 1) & (y == 0)).sum()),
            "fn": int(((yhat == 0) & (y == 1)).sum()), "tn": int(((yhat == 0) & (y == 0)).sum())}


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def subset(block: dict, keep) -> tuple[list, list]:
    idx = [i for i, sid in enumerate(block["ids"]) if sid in keep]
    return [block["y"][i] for i in idx], [block["p"][i] for i in idx]


def run_metrics(path: Path, dataset: str) -> dict:
    with gzip.open(path, "rt") as f:
        r = json.load(f)
    thr, test = r["threshold"], r["test"]
    out = {"run": r["meta"]["run"], "threshold": thr,
           "all": {k: scores(v["y"], v["p"], thr) for k, v in test.items()}, "valid": {}}
    for c in CONDITIONS:
        if c not in test:
            continue
        keep = valid_ids(dataset, c) & set(test[c]["ids"]) & set(test["clean"]["ids"])
        out["valid"][c] = {"transformed": scores(*subset(test[c], keep), thr),
                           "clean": scores(*subset(test["clean"], keep), thr)}
    return out


def summarize(dataset: str, model: str) -> dict:
    runs = sorted((ROOT / "results" / "preds" / dataset / model).glob("run_*.json.gz"))
    per_run = [run_metrics(p, dataset) for p in runs]
    agg = {"dataset": dataset, "model": model, "n_runs": len(per_run), "per_run": per_run}
    for scope in ("all", "valid"):
        agg[scope] = {}
        for c in CONDITIONS:
            if scope == "all":
                cl = [r["all"]["clean"] for r in per_run]
                tr = [r["all"][c] for r in per_run if c in r["all"]]
            else:
                cl = [r["valid"][c]["clean"] for r in per_run if c in r["valid"]]
                tr = [r["valid"][c]["transformed"] for r in per_run if c in r["valid"]]
            if not tr:
                continue
            row = {}
            for m in ("f1", "precision", "recall", "pr_auc", "roc_auc", "brier", "ece"):
                d = np.array([t[m] - k[m] for t, k in zip(tr, cl)])
                row[m] = {"clean_mean": float(np.mean([k[m] for k in cl])),
                          "clean_std": float(np.std([k[m] for k in cl], ddof=1)) if len(cl) > 1 else 0.0,
                          "delta_mean": float(d.mean()), "delta_per_run": d.round(4).tolist()}
            row["n"] = tr[0]["n"]
            row["n_vul"] = tr[0]["n_vul"]
            agg[scope][c] = row
    out = ROOT / "results" / "metrics" / dataset / f"{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=1))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    agg = summarize(a.dataset, a.model)
    print(f"{a.model} on {a.dataset}: {agg['n_runs']} runs")
    for scope in ("all", "valid"):
        row = agg[scope]
        if not row:
            continue
        first = next(iter(row.values()))["f1"]
        print(f"  [{scope}] clean F1 {first['clean_mean']:.2f}±{first['clean_std']:.2f} | " + "  ".join(
            f"Δ{c}={v['f1']['delta_mean']:+.2f}(n={v['n']})" for c, v in row.items()))


if __name__ == "__main__":
    main()
