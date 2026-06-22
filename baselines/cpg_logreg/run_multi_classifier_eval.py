#!/usr/bin/env python3
"""
Experiment A: Multiple classifiers on mean-pooled CPG features.

Tests whether the robustness pattern (identifier renaming hurts most,
control flow barely matters) is consistent across classifier types —
demonstrating it is feature-driven, not model-specific.

Classifiers: LogReg, SVM (RBF), Random Forest, MLP
Data: devign_full/after_ggnn_meanpool/ (200-dim mean-pooled CPG node features)
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

THESIS_ROOT = Path(__file__).resolve().parents[2]
MEANPOOL_DIR = THESIS_ROOT / "devign_full" / "after_ggnn_meanpool"
OUT_PATH     = THESIS_ROOT / "devign_full" / "multi_classifier_results.json"

N_TRIALS   = 30
SUBSAMPLE  = 5000
SEED_BASE  = 42

CONDITIONS = [
    ("train",           "train_GGNNinput_graph.json",                    "Training set"),
    ("test_originals",  "test_GGNNinput_graph.json",                     "Originals test"),
    ("test_identifier", "obf_identifier_test_GGNNinput_graph.json",      "Obf: identifier renaming"),
    ("test_deadcode",   "obf_deadcode_test_GGNNinput_graph.json",        "Obf: dead code insertion"),
    ("test_controlflow","obf_controlflow_test_GGNNinput_graph.json",     "Obf: control flow obfuscation"),
]

CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(C=1.0, class_weight="balanced",
                                          max_iter=1000, random_state=0),
    "svm_rbf": lambda: SVC(C=1.0, kernel="rbf", class_weight="balanced",
                            probability=False, random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100,
                                                     class_weight="balanced",
                                                     n_jobs=-1, random_state=0),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300,
                                  early_stopping=True, random_state=0),
}


def load_split(fname):
    path = MEANPOOL_DIR / fname
    data = json.load(open(path))
    X = np.array([r["graph_feature"] for r in data], dtype=np.float32)
    y = np.array([int(r["target"]) for r in data], dtype=np.int32)
    return X, y


def evaluate_clf(clf, X, y):
    preds = clf.predict(X)
    return (accuracy_score(y, preds) * 100,
            precision_score(y, preds, zero_division=0) * 100,
            recall_score(y, preds, zero_division=0) * 100,
            f1_score(y, preds, zero_division=0) * 100)


def run_trials(clf_name, clf_factory, X_train_full, y_train_full, test_splits):
    results = {k: {"acc": [], "pr": [], "rc": [], "f1": []}
               for k in test_splits}

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(SEED_BASE + trial)
        # Stratified subsample of train
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]
        n_pos = int(SUBSAMPLE * len(pos_idx) / len(y_train_full))
        n_neg = SUBSAMPLE - n_pos
        sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(idx)
        X_tr, y_tr = X_train_full[idx], y_train_full[idx]

        clf = clf_factory()
        # MLP doesn't support class_weight; use sample_weight
        if clf_name == "mlp":
            sw = compute_sample_weight("balanced", y_tr)
            clf.fit(X_tr, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr, y_tr)

        for cond_key, (X_te, y_te) in test_splits.items():
            acc, pr, rc, f1 = evaluate_clf(clf, X_te, y_te)
            results[cond_key]["acc"].append(acc)
            results[cond_key]["pr"].append(pr)
            results[cond_key]["rc"].append(rc)
            results[cond_key]["f1"].append(f1)

        if (trial + 1) % 5 == 0:
            baseline_f1 = np.mean(results["test_originals"]["f1"])
            print(f"  [{clf_name}] Trial {trial+1}/{N_TRIALS}  "
                  f"baseline F1={baseline_f1:.2f}%", flush=True)

    return {k: {
        "acc_mean":  float(np.mean(v["acc"])),  "acc_std":  float(np.std(v["acc"])),
        "pr_mean":   float(np.mean(v["pr"])),   "pr_std":   float(np.std(v["pr"])),
        "rc_mean":   float(np.mean(v["rc"])),   "rc_std":   float(np.std(v["rc"])),
        "f1_mean":   float(np.mean(v["f1"])),   "f1_std":   float(np.std(v["f1"])),
    } for k, v in results.items()}


def print_table(all_results):
    cond_keys = ["test_originals", "test_identifier", "test_deadcode", "test_controlflow"]
    cond_labels = ["Originals", "Identifier renaming", "Dead code insertion", "Control flow obf."]
    clf_names = list(all_results.keys())

    print("\n" + "=" * 80)
    print("ROBUSTNESS RESULTS — Mean F1 ± std (30 trials, 5000-sample subsample)")
    print("=" * 80)
    header = f"{'Condition':<28}" + "".join(f"{'  ' + n:<20}" for n in clf_names)
    print(header)
    print("-" * 80)
    for ckey, clabel in zip(cond_keys, cond_labels):
        row = f"{clabel:<28}"
        for clf in clf_names:
            m = all_results[clf][ckey]["f1_mean"]
            s = all_results[clf][ckey]["f1_std"]
            row += f"  {m:5.2f}±{s:4.2f}      "
        print(row)
    print("-" * 80)
    # Delta rows
    print(f"{'Δ F1 vs originals:':<28}")
    for ckey, clabel in zip(cond_keys[1:], cond_labels[1:]):
        row = f"  {clabel:<26}"
        for clf in clf_names:
            base = all_results[clf]["test_originals"]["f1_mean"]
            delta = all_results[clf][ckey]["f1_mean"] - base
            row += f"  {delta:+6.2f}           "
        print(row)
    print("=" * 80)


def main():
    print("Loading mean-pooled CPG features…", flush=True)
    X_train, y_train = load_split("train_GGNNinput_graph.json")
    print(f"  train: {len(y_train)} samples  "
          f"pos={y_train.sum()}  neg={(y_train==0).sum()}", flush=True)

    test_splits = {}
    for cond_key, fname, label in CONDITIONS[1:]:  # skip train
        X, y = load_split(fname)
        test_splits[cond_key] = (X, y)
        print(f"  {cond_key}: {len(y)} samples  pos={y.sum()}", flush=True)

    all_results = {}
    for clf_name, clf_factory in CLASSIFIERS.items():
        print(f"\nRunning {clf_name} ({N_TRIALS} trials)…", flush=True)
        all_results[clf_name] = run_trials(
            clf_name, clf_factory, X_train, y_train, test_splits)

    print_table(all_results)

    # Save results
    out = {
        "n_trials": N_TRIALS,
        "subsample": SUBSAMPLE,
        "feature_source": "mean-pooled CPG node features (200-dim)",
        "classifiers": all_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
