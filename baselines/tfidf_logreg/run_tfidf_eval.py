#!/usr/bin/env python3
"""
Experiment B: TF-IDF on raw C source code + Logistic Regression.

Compares CPG-based features (Experiment A) with a pure text baseline that
has no knowledge of program structure. Tests whether robustness patterns
are CPG-specific or appear in any token-based representation.

Features: TF-IDF (word unigrams+bigrams, max 10k features, sublinear_tf)
Classifier: Logistic Regression (class_weight='balanced')
Data: originals_full_data_with_slices.json + obf variant files
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.sparse import issparse

THESIS_ROOT  = Path(__file__).resolve().parents[2]
DEVIGN_FULL  = THESIS_ROOT / "devign_full"
SPLIT_FILE   = DEVIGN_FULL / "devign_full_split_801010.json"
ORIG_FILE    = DEVIGN_FULL / "originals_full_data_with_slices.json"
OBF_FILES    = {
    "test_identifier":  DEVIGN_FULL / "obf_identifier_full_data_with_slices.json",
    "test_deadcode":    DEVIGN_FULL / "obf_deadcode_full_data_with_slices.json",
    "test_controlflow": DEVIGN_FULL / "obf_controlflow_full_data_with_slices.json",
}
OUT_PATH = DEVIGN_FULL / "tfidf_results.json"

N_TRIALS  = 30
SUBSAMPLE = 5000
SEED_BASE = 42

TFIDF_PARAMS = dict(
    max_features=10000,
    sublinear_tf=True,
    ngram_range=(1, 2),
    analyzer="word",
    token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",   # C identifiers + keywords
    min_df=3,
)


def load_source_split():
    """Load source code aligned with the 80/10/10 split."""
    split = json.load(open(SPLIT_FILE))
    train_names = set(split["splits"]["train"])
    test_names  = set(split["splits"]["test"])

    orig = json.load(open(ORIG_FILE))
    fn_map = {r["file_name"]: r for r in orig}

    train_texts, train_labels = [], []
    for n in split["splits"]["train"]:   # preserve split order
        if n in fn_map:
            train_texts.append(fn_map[n]["code"])
            train_labels.append(int(fn_map[n]["label"]))

    test_texts, test_labels = [], []
    for n in split["splits"]["test"]:
        if n in fn_map:
            test_texts.append(fn_map[n]["code"])
            test_labels.append(int(fn_map[n]["label"]))

    obf_data = {}
    for cond_key, obf_path in OBF_FILES.items():
        obf_map = {r["file_name"]: r for r in json.load(open(obf_path))}
        texts, labels = [], []
        for n in split["splits"]["test"]:
            if n in obf_map:
                texts.append(obf_map[n]["code"])
                labels.append(int(obf_map[n]["label"]))
        obf_data[cond_key] = (texts, labels)

    return (train_texts, np.array(train_labels, dtype=np.int32),
            test_texts,  np.array(test_labels, dtype=np.int32),
            obf_data)


def evaluate_clf(clf, X, y):
    preds = clf.predict(X)
    return (accuracy_score(y, preds) * 100,
            precision_score(y, preds, zero_division=0) * 100,
            recall_score(y, preds, zero_division=0) * 100,
            f1_score(y, preds, zero_division=0) * 100)


def run_trials(X_train_full, y_train_full, X_test, y_test, obf_splits):
    cond_keys = ["test_originals"] + list(obf_splits.keys())
    results   = {k: {"acc": [], "pr": [], "rc": [], "f1": []} for k in cond_keys}

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(SEED_BASE + trial)
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]
        n_pos = int(SUBSAMPLE * len(pos_idx) / len(y_train_full))
        n_neg = SUBSAMPLE - n_pos
        sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(idx)

        X_tr = X_train_full[idx]
        y_tr = y_train_full[idx]

        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000, random_state=trial)
        clf.fit(X_tr, y_tr)

        acc, pr, rc, f1 = evaluate_clf(clf, X_test, y_test)
        results["test_originals"]["acc"].append(acc)
        results["test_originals"]["pr"].append(pr)
        results["test_originals"]["rc"].append(rc)
        results["test_originals"]["f1"].append(f1)

        for cond_key, (X_obf, y_obf) in obf_splits.items():
            acc, pr, rc, f1 = evaluate_clf(clf, X_obf, y_obf)
            results[cond_key]["acc"].append(acc)
            results[cond_key]["pr"].append(pr)
            results[cond_key]["rc"].append(rc)
            results[cond_key]["f1"].append(f1)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS}  "
                  f"F1(originals)={np.mean(results['test_originals']['f1']):.2f}%",
                  flush=True)

    return {k: {
        "acc_mean": float(np.mean(v["acc"])), "acc_std": float(np.std(v["acc"])),
        "pr_mean":  float(np.mean(v["pr"])),  "pr_std":  float(np.std(v["pr"])),
        "rc_mean":  float(np.mean(v["rc"])),  "rc_std":  float(np.std(v["rc"])),
        "f1_mean":  float(np.mean(v["f1"])),  "f1_std":  float(np.std(v["f1"])),
    } for k, v in results.items()}


def main():
    print("Loading source code splits…", flush=True)
    (train_texts, y_train,
     test_texts, y_test,
     obf_data) = load_source_split()

    print(f"  train: {len(y_train)}  pos={y_train.sum()}  neg={(y_train==0).sum()}",
          flush=True)
    print(f"  test:  {len(y_test)}   pos={y_test.sum()}", flush=True)
    for k, (txts, lbls) in obf_data.items():
        print(f"  {k}: {len(lbls)}", flush=True)

    print("\nFitting TF-IDF vectorizer on train corpus…", flush=True)
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    obf_splits = {k: (vec.transform(txts), np.array(lbls, dtype=np.int32))
                  for k, (txts, lbls) in obf_data.items()}

    print(f"  Vocab size: {len(vec.vocabulary_)}", flush=True)
    print(f"  X_train shape: {X_train.shape}", flush=True)

    y_train = np.array(y_train, dtype=np.int32)

    print(f"\nRunning {N_TRIALS} trials…", flush=True)
    results = run_trials(X_train, y_train, X_test, y_test, obf_splits)

    cond_labels = {
        "test_originals":  "Originals test",
        "test_identifier": "Obf: identifier renaming",
        "test_deadcode":   "Obf: dead code insertion",
        "test_controlflow":"Obf: control flow obfuscation",
    }

    print("\n" + "=" * 65)
    print("TF-IDF + LogReg  —  F1 mean ± std (30 trials, 5000-subsample)")
    print("=" * 65)
    baseline = results["test_originals"]["f1_mean"]
    for k, label in cond_labels.items():
        m, s = results[k]["f1_mean"], results[k]["f1_std"]
        delta = m - baseline if k != "test_originals" else 0.0
        sign = f"  Δ{delta:+.2f}%" if k != "test_originals" else ""
        print(f"  {label:<34}  F1={m:.2f}±{s:.2f}%{sign}")
    print("=" * 65)

    out = {
        "n_trials": N_TRIALS,
        "subsample": SUBSAMPLE,
        "feature_source": "TF-IDF on raw C source (word 1-2grams, max 10k, sublinear_tf)",
        "vocab_size": len(vec.vocabulary_),
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
