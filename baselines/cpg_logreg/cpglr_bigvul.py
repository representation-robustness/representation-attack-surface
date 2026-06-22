#!/usr/bin/env python3
"""
CPG + Logistic Regression on Big-Vul.

Extracts graph-level features from BigVul CPG pkl files:
  - Node type distribution (NUM_NODE_TYPES bins)
  - Edge type distribution (NUM_EDGE_TYPES bins)
  - Graph-level statistics (num_nodes, num_edges, density)

Runs 30 trials (subsampled) and evaluates on 4 test splits.
Outputs: ~/thesis/devign_full/bigvul_cpglr_results.json
"""

import json, os, sys, random
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.expanduser("~"))
from bigvul_cpg_parser import load_split
from bigvul_cpg_parser import NUM_NODE_TYPES, NUM_EDGE_TYPES

RESULTS_DIR = Path(os.path.expanduser("~/thesis/devign_full"))
N_TRIALS    = 30
SUBSAMPLE   = 5000
SEED_BASE   = 42

SPLITS = {
    "test":                 "test",
    "test_obf_identifier":  "test_obf_identifier",
    "test_obf_deadcode":    "test_obf_deadcode",
    "test_obf_controlflow": "test_obf_controlflow",
}


def graph_to_features(g):
    x = g.x.squeeze(-1).long()
    e = g.edge_attr.long() if g.edge_attr is not None else None

    node_hist = np.zeros(NUM_NODE_TYPES + 1)
    for ntype in x.numpy():
        node_hist[min(int(ntype), NUM_NODE_TYPES)] += 1
    node_hist = node_hist / (node_hist.sum() + 1e-9)

    edge_hist = np.zeros(NUM_EDGE_TYPES)
    if e is not None and len(e) > 0:
        for etype in e.numpy():
            edge_hist[min(int(etype), NUM_EDGE_TYPES - 1)] += 1
    n_edges = edge_hist.sum()
    edge_hist = edge_hist / (n_edges + 1e-9)

    n_nodes = len(x)
    density = n_edges / (n_nodes * (n_nodes - 1) + 1e-9)
    stats = np.array([np.log1p(n_nodes), np.log1p(n_edges), density])

    return np.concatenate([node_hist, edge_hist, stats])


def extract_features(graphs):
    X = np.array([graph_to_features(g) for g in graphs])
    y = np.array([int(g.y.item()) for g in graphs])
    return X, y


def run_trials(X_train, y_train, X_tests, n_trials, subsample):
    results = {k: [] for k in X_tests}
    rng = np.random.RandomState(SEED_BASE)

    for trial in range(n_trials):
        idx = rng.choice(len(X_train), size=min(subsample, len(X_train)), replace=False)
        Xs, ys = X_train[idx], y_train[idx]

        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1000, random_state=trial)
        clf.fit(Xs, ys)

        for split_name, (Xt, yt) in X_tests.items():
            preds = clf.predict(Xt)
            results[split_name].append(f1_score(yt, preds, zero_division=0) * 100)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials} done", flush=True)

    return results


def main():
    print("Loading BigVul CPG graphs...", flush=True)
    train_graphs = load_split("train")
    print(f"  train={len(train_graphs)}", flush=True)

    test_graphs = {}
    for split_name, split_key in SPLITS.items():
        test_graphs[split_name] = load_split(split_key)
        print(f"  {split_name}={len(test_graphs[split_name])}", flush=True)

    print("Extracting features...", flush=True)
    X_train, y_train = extract_features(train_graphs)
    X_tests = {k: extract_features(v) for k, v in test_graphs.items()}

    print(f"Feature dim: {X_train.shape[1]}", flush=True)
    print(f"Running {N_TRIALS} trials (subsample={SUBSAMPLE})...", flush=True)

    trial_results = run_trials(X_train, y_train, X_tests, N_TRIALS, SUBSAMPLE)

    base = float(np.mean(trial_results["test"]))
    agg = {}
    for split_name, f1s in trial_results.items():
        agg[split_name] = {
            "f1_mean":  round(float(np.mean(f1s)), 2),
            "f1_std":   round(float(np.std(f1s)), 2),
            "all_f1":   [round(v, 2) for v in f1s],
        }
    for split_name in list(SPLITS.keys())[1:]:
        agg[split_name]["delta_f1"] = round(agg[split_name]["f1_mean"] - base, 2)

    agg.update({"model": "CPG+LR", "dataset": "Big-Vul",
                "n_trials": N_TRIALS, "subsample": SUBSAMPLE})

    out = RESULTS_DIR / "bigvul_cpglr_results.json"
    with open(out, "w") as f: json.dump(agg, f, indent=2)
    print(f"\nResults → {out}", flush=True)
    print(f"  test: F1={agg['test']['f1_mean']:.2f}±{agg['test']['f1_std']:.2f}%")
    for k in list(SPLITS.keys())[1:]:
        d = agg[k]
        print(f"  {k}: F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}% Δ={d['delta_f1']:+.2f}pp")


if __name__ == "__main__":
    main()
