#!/usr/bin/env python3
"""TF-IDF + LogReg on Devign, all 7 obfuscation conditions (30 trials)."""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

THESIS_ROOT = Path(__file__).resolve().parents[2]
DEVIGN_FULL = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_FULL / "devign_full_split_801010.json"
ORIG_FILE   = DEVIGN_FULL / "originals_full_data_with_slices.json"
OBF_FILES   = {
    "test_obf_identifier":  DEVIGN_FULL / "obf_identifier_full_data_with_slices.json",
    "test_obf_deadcode":    DEVIGN_FULL / "obf_deadcode_full_data_with_slices.json",
    "test_obf_controlflow": DEVIGN_FULL / "obf_controlflow_full_data_with_slices.json",
    "test_obf_ren_dead":    DEVIGN_FULL / "obf_ren_dead_full_data_with_slices.json",
    "test_obf_ren_cf":      DEVIGN_FULL / "obf_ren_cf_full_data_with_slices.json",
    "test_obf_dead_cf":     DEVIGN_FULL / "obf_dead_cf_full_data_with_slices.json",
    "test_obf_compound":    DEVIGN_FULL / "obf_compound_full_data_with_slices.json",
}
OUT_PATH = DEVIGN_FULL / "tfidf_7cond_devign_results.json"

N_TRIALS  = 30
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


def load_splits():
    split  = json.load(open(SPLIT_FILE))
    orig   = json.load(open(ORIG_FILE))
    fn_map = {r["file_name"]: r for r in orig}

    train_texts, train_labels = [], []
    for n in split["splits"]["train"]:
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
        print(f"  {cond_key}: {len(labels)} samples", flush=True)

    return (train_texts, np.array(train_labels, dtype=np.int32),
            test_texts, np.array(test_labels, dtype=np.int32),
            obf_data)


def run_trials(X_train_full, y_train_full, X_test, y_test, obf_splits):
    all_keys  = ["test"] + list(obf_splits.keys())
    f1_lists  = {k: [] for k in all_keys}

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
                                 max_iter=1000, random_state=trial)
        clf.fit(X_train_full[idx], y_train_full[idx])

        f1_lists["test"].append(f1_score(y_test, clf.predict(X_test),
                                         zero_division=0) * 100)
        for k, (X_obf, y_obf) in obf_splits.items():
            f1_lists[k].append(f1_score(y_obf, clf.predict(X_obf),
                                        zero_division=0) * 100)

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS}  "
                  f"F1(test)={np.mean(f1_lists['test']):.2f}%", flush=True)

    return f1_lists


def main():
    print("Loading splits...", flush=True)
    train_texts, y_train, test_texts, y_test, obf_data = load_splits()
    print(f"  train: {len(y_train)}, test: {len(y_test)}", flush=True)

    print("Fitting TF-IDF...", flush=True)
    vec     = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    obf_splits = {k: (vec.transform(txts), np.array(lbls, dtype=np.int32))
                  for k, (txts, lbls) in obf_data.items()}
    print(f"  Vocab: {len(vec.vocabulary_)}", flush=True)

    print(f"\nRunning {N_TRIALS} trials...", flush=True)
    f1_lists = run_trials(X_train, y_train, X_test, y_test, obf_splits)

    base = np.mean(f1_lists["test"])
    out  = {"test": {
        "f1_mean": round(float(base), 2),
        "f1_std":  round(float(np.std(f1_lists["test"])), 2),
        "all_f1":  [round(v, 2) for v in f1_lists["test"]],
    }}
    for k, vals in f1_lists.items():
        if k == "test":
            continue
        out[k] = {
            "f1_mean":  round(float(np.mean(vals)), 2),
            "f1_std":   round(float(np.std(vals)), 2),
            "all_f1":   [round(v, 2) for v in vals],
            "delta_f1": round(float(np.mean(vals)) - float(base), 2),
        }

    print("\n=== RESULTS ===")
    print(f"{'Condition':<30} {'F1':>8} {'StD':>8} {'ΔF1':>8}")
    for k, v in out.items():
        delta = f"{v.get('delta_f1', 0):+.2f}" if k != "test" else "—"
        print(f"{k:<30} {v['f1_mean']:>7.2f}% {v['f1_std']:>7.2f}% {delta:>8}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
