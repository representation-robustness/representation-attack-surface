#!/usr/bin/env python3
"""CPG + Logistic Regression on DiverseVul — all 7 obfuscation conditions."""
import json, os, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.expanduser("~"))
from diversevul_cpg_parser import load_split
from diversevul_cpg_parser import NUM_NODE_TYPES, NUM_EDGE_TYPES

RESULTS_DIR = Path(os.path.expanduser("~/thesis/devign_full"))
N_TRIALS    = 30
SUBSAMPLE   = 5000
SEED_BASE   = 42

SPLITS = {
    "test":                  "test",
    "test_obf_identifier":   "test_obf_identifier",
    "test_obf_deadcode":     "test_obf_deadcode",
    "test_obf_controlflow":  "test_obf_controlflow",
    "test_obf_ren_dead":     "test_obf_ren_dead",
    "test_obf_ren_cf":       "test_obf_ren_cf",
    "test_obf_dead_cf":      "test_obf_dead_cf",
    "test_obf_compound":     "test_obf_compound",
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


def main():
    print("Loading DiverseVul CPG graphs...", flush=True)
    train_graphs = load_split("train")
    X_train, y_train = extract_features(train_graphs)
    print(f"  train: {len(y_train)}, feat_dim={X_train.shape[1]}", flush=True)

    X_tests = {}
    for split_name, split_key in SPLITS.items():
        graphs = load_split(split_key)
        X, y = extract_features(graphs)
        X_tests[split_name] = (X, y)
        print(f"  {split_name}: {len(y)}", flush=True)

    print(f"\nRunning {N_TRIALS} trials (subsample={SUBSAMPLE})...", flush=True)
    f1_lists = {k: [] for k in X_tests}
    rng = np.random.RandomState(SEED_BASE)

    for trial in range(N_TRIALS):
        idx = rng.choice(len(X_train), size=min(SUBSAMPLE, len(X_train)), replace=False)
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1000, random_state=trial)
        clf.fit(X_train[idx], y_train[idx])
        for k, (Xt, yt) in X_tests.items():
            f1_lists[k].append(f1_score(yt, clf.predict(Xt), zero_division=0) * 100)
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS}  "
                  f"F1(test)={np.mean(f1_lists['test']):.2f}%", flush=True)

    base = float(np.mean(f1_lists["test"]))
    out = {"test": {
        "f1_mean": round(base, 2),
        "f1_std":  round(float(np.std(f1_lists["test"])), 2),
        "all_f1":  [round(v, 2) for v in f1_lists["test"]],
    }}
    for k, vals in f1_lists.items():
        if k == "test":
            continue
        m = float(np.mean(vals))
        out[k] = {
            "f1_mean":  round(m, 2),
            "f1_std":   round(float(np.std(vals)), 2),
            "all_f1":   [round(v, 2) for v in vals],
            "delta_f1": round(m - base, 2),
        }
    out.update({"model": "CPG+LR", "dataset": "DiverseVul",
                "n_trials": N_TRIALS, "subsample": SUBSAMPLE})

    print("\n=== RESULTS ===")
    KEYS = ["test","test_obf_identifier","test_obf_deadcode","test_obf_controlflow",
            "test_obf_ren_dead","test_obf_ren_cf","test_obf_dead_cf","test_obf_compound"]
    for k in KEYS:
        v = out[k]
        delta = f"{v.get('delta_f1',0):+.2f}" if k != "test" else "—"
        print(f"  {k:<30} {v['f1_mean']:>7.2f}% {v['f1_std']:>7.2f}%  Δ={delta}")

    result_path = RESULTS_DIR / "cpglr_7cond_diversevul_results.json"
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {result_path}", flush=True)


if __name__ == "__main__":
    main()
