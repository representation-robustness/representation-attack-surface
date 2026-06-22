"""
bigvul_tfidf_eval.py — TF-IDF + LR on Big-Vul (cross-dataset generalization).

Runs after bigvul_preprocess.py has created the splits.
Uses the same TF-IDF config as our Devign experiment (fair comparison).

Usage:
    python bigvul_tfidf_eval.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SPLITS_DIR  = Path.home() / "thesis/bigvul/splits"
RESULTS_DIR = Path.home() / "thesis/devign_full"

N_TRIALS  = 10
SUBSAMPLE = 5000
SEED_BASE = 42

TFIDF_PARAMS = dict(
    max_features=10000,
    sublinear_tf=True,
    ngram_range=(1, 2),
    analyzer="word",
    token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
    min_df=3,
)

CONDITIONS = {
    "original":    "test.jsonl",
    "identifier":  "test_obf_identifier.jsonl",
    "deadcode":    "test_obf_deadcode.jsonl",
    "controlflow": "test_obf_controlflow.jsonl",
}


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    texts  = [r['func'] for r in rows]
    labels = np.array([r['target'] for r in rows], dtype=int)
    return texts, labels


def main():
    print("=== TF-IDF + LR on Big-Vul ===\n")

    train_file = SPLITS_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"ERROR: {train_file} not found. Run bigvul_preprocess.py first.")
        return

    train_texts, train_labels = load_jsonl(train_file)
    print(f"Train: {len(train_texts)} samples (vul={train_labels.sum()}, clean={(train_labels==0).sum()})")

    results = {}
    for cond, fname in CONDITIONS.items():
        fpath = SPLITS_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {cond}: {fpath} not found")
            continue

        test_texts, test_labels = load_jsonl(fpath)
        print(f"\n  {cond}: {len(test_texts)} test samples")

        all_f1, all_pr, all_rc, all_acc = [], [], [], []
        for trial in range(N_TRIALS):
            seed = SEED_BASE + trial
            rng  = np.random.RandomState(seed)
            idx  = rng.choice(len(train_texts), min(SUBSAMPLE, len(train_texts)), replace=False)
            sub_texts  = [train_texts[i] for i in idx]
            sub_labels = train_labels[idx]

            vec = TfidfVectorizer(**TFIDF_PARAMS)
            X_tr = vec.fit_transform(sub_texts)
            X_te = vec.transform(test_texts)

            clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                     C=1.0, random_state=seed)
            clf.fit(X_tr, sub_labels)
            preds = clf.predict(X_te)

            all_f1.append( f1_score(test_labels,  preds, zero_division=0) * 100)
            all_pr.append( precision_score(test_labels, preds, zero_division=0) * 100)
            all_rc.append( recall_score(test_labels,    preds, zero_division=0) * 100)
            all_acc.append(accuracy_score(test_labels,  preds) * 100)

        results[cond] = {
            "f1_mean":  round(float(np.mean(all_f1)), 2),
            "f1_std":   round(float(np.std(all_f1)), 2),
            "pr_mean":  round(float(np.mean(all_pr)), 2),
            "rc_mean":  round(float(np.mean(all_rc)), 2),
            "acc_mean": round(float(np.mean(all_acc)), 2),
            "all_f1":   [round(x, 2) for x in all_f1],
        }
        print(f"    F1={results[cond]['f1_mean']:.2f} ± {results[cond]['f1_std']:.2f}%  "
              f"Pr={results[cond]['pr_mean']:.2f}%  Rc={results[cond]['rc_mean']:.2f}%")

    base = results.get("original", {}).get("f1_mean", 0)
    for cond in results:
        if cond != "original":
            results[cond]["delta_f1"] = round(results[cond]["f1_mean"] - base, 2)

    out = {
        "dataset": "Big-Vul (MSR 2021)",
        "model": "TF-IDF + LR",
        "n_trials": N_TRIALS,
        "subsample_per_trial": SUBSAMPLE,
        "tfidf_params": TFIDF_PARAMS,
        "results": results,
    }
    out_path = RESULTS_DIR / "bigvul_tfidf_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Big-Vul TF-IDF Summary ===")
    for cond, m in results.items():
        delta = f"  ΔF1={m.get('delta_f1', 0):+.2f}" if cond != 'original' else ''
        print(f"  {cond:<20} F1={m['f1_mean']:.2f} ± {m['f1_std']:.2f}%{delta}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
