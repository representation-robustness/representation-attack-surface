#!/usr/bin/env python3
"""
CPG+LR eval for compound (dead+cf) condition on Devign.

Since renaming doesn't affect CPG node-type features:
  ΔD+CF = ΔCmp (compound = ren+dead+cf, but ren has zero effect)
  ΔR+D  = ΔDead = +6.11  (from existing multi_classifier_results.json)
  ΔR+CF = ΔCF  = -1.11

Outputs full 7-condition CPG+LR results.json with all pairwise cells filled.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

THESIS_ROOT = Path(__file__).resolve().parent
MEANPOOL_DIR = THESIS_ROOT / "devign_full" / "after_ggnn_meanpool"
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"
EXISTING     = THESIS_ROOT / "devign_full" / "multi_classifier_results.json"
OUT_PATH     = THESIS_ROOT / "devign_full" / "cpglr_7cond_devign_results.json"

N_TRIALS  = 30
SUBSAMPLE = 5000
SEED_BASE = 42
HIDDEN_DIM = 200


def mean_pool_pad(node_features):
    arr = np.array(node_features, dtype=np.float32)
    mean_vec = arr.mean(axis=0)
    if mean_vec.shape[0] < HIDDEN_DIM:
        pad = np.zeros(HIDDEN_DIM - mean_vec.shape[0], dtype=np.float32)
        mean_vec = np.concatenate([mean_vec, pad])
    return mean_vec[:HIDDEN_DIM]


def load_meanpool_json(path):
    data = json.load(open(path))
    X = np.array([r["graph_feature"] for r in data], dtype=np.float32)
    y = np.array([int(r["target"]) for r in data], dtype=np.int32)
    return X, y


def load_ggnn_input(path):
    records = json.load(open(path))
    X = np.array([mean_pool_pad(r["node_features"]) for r in records], dtype=np.float32)
    y = np.array([int(r["targets"][0][0]) for r in records], dtype=np.int32)
    return X, y


def run_30trials(X_train_full, y_train_full, test_splits):
    f1_lists = {k: [] for k in test_splits}
    for trial in range(N_TRIALS):
        rng   = np.random.default_rng(SEED_BASE + trial)
        pos   = np.where(y_train_full == 1)[0]
        neg   = np.where(y_train_full == 0)[0]
        n_pos = int(SUBSAMPLE * len(pos) / len(y_train_full))
        n_neg = SUBSAMPLE - n_pos
        idx   = np.concatenate([
            rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
        ])
        rng.shuffle(idx)
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1000, random_state=0)
        clf.fit(X_train_full[idx], y_train_full[idx])
        for k, (X_te, y_te) in test_splits.items():
            f1_lists[k].append(f1_score(y_te, clf.predict(X_te),
                                        zero_division=0) * 100)
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS}  "
                  f"F1(test)={np.mean(f1_lists['test']):.2f}%", flush=True)
    return f1_lists


def main():
    print("Loading train features...", flush=True)
    X_train, y_train = load_meanpool_json(MEANPOOL_DIR / "train_GGNNinput_graph.json")
    print(f"  train: {len(y_train)}", flush=True)

    print("Loading test features...", flush=True)
    X_test, y_test   = load_meanpool_json(MEANPOOL_DIR / "test_GGNNinput_graph.json")
    X_iden, y_iden   = load_meanpool_json(MEANPOOL_DIR / "obf_identifier_test_GGNNinput_graph.json")
    X_dead, y_dead   = load_meanpool_json(MEANPOOL_DIR / "obf_deadcode_test_GGNNinput_graph.json")
    X_cf,   y_cf     = load_meanpool_json(MEANPOOL_DIR / "obf_controlflow_test_GGNNinput_graph.json")

    print("Loading compound (dead+cf) from GGNNinput.json...", flush=True)
    X_cmp, y_cmp = load_ggnn_input(DEVIGN_INPUT / "obf_compound_test" / "test_GGNNinput.json")
    print(f"  compound: {len(y_cmp)}", flush=True)

    test_splits = {
        "test":                 (X_test, y_test),
        "test_obf_identifier":  (X_iden, y_iden),
        "test_obf_deadcode":    (X_dead, y_dead),
        "test_obf_controlflow": (X_cf,   y_cf),
        "test_obf_compound":    (X_cmp,  y_cmp),
    }

    print(f"\nRunning {N_TRIALS} trials...", flush=True)
    f1_lists = run_30trials(X_train, y_train, test_splits)

    base = float(np.mean(f1_lists["test"]))
    out  = {"test": {
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

    # ΔR+D = ΔDead (renaming doesn't change node-type features)
    dead = out["test_obf_deadcode"]["f1_mean"]
    cf   = out["test_obf_controlflow"]["f1_mean"]
    cmp  = out["test_obf_compound"]["f1_mean"]

    out["test_obf_ren_dead"] = {
        "f1_mean":  dead,
        "f1_std":   out["test_obf_deadcode"]["f1_std"],
        "all_f1":   out["test_obf_deadcode"]["all_f1"],
        "delta_f1": round(dead - base, 2),
        "note":     "identical to dead-code (renaming-invariant node features)",
    }
    out["test_obf_ren_cf"] = {
        "f1_mean":  cf,
        "f1_std":   out["test_obf_controlflow"]["f1_std"],
        "all_f1":   out["test_obf_controlflow"]["all_f1"],
        "delta_f1": round(cf - base, 2),
        "note":     "identical to control-flow (renaming-invariant node features)",
    }
    out["test_obf_dead_cf"] = {
        "f1_mean":  cmp,
        "f1_std":   out["test_obf_compound"]["f1_std"],
        "all_f1":   out["test_obf_compound"]["all_f1"],
        "delta_f1": round(cmp - base, 2),
        "note":     "identical to compound (renaming-invariant node features)",
    }

    out["n_trials"] = N_TRIALS
    out["model"]    = "CPG+LR"
    out["dataset"]  = "Devign"

    print("\n=== RESULTS ===")
    print(f"{'Condition':<30} {'F1':>8} {'StD':>8} {'ΔF1':>8}")
    KEYS = ["test", "test_obf_identifier", "test_obf_deadcode", "test_obf_controlflow",
            "test_obf_ren_dead", "test_obf_ren_cf", "test_obf_dead_cf", "test_obf_compound"]
    for k in KEYS:
        if k not in out:
            continue
        v = out[k]
        delta = f"{v.get('delta_f1', 0):+.2f}" if k != "test" else "—"
        print(f"{k:<30} {v['f1_mean']:>7.2f}% {v['f1_std']:>7.2f}% {delta:>8}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
