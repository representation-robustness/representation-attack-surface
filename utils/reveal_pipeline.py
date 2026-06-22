"""
reveal_pipeline.py — Prepare ReVeal dataset splits + obfuscation + TF-IDF eval.

Source: ~/thesis/data/raw/ReVeal/data/function.json
        27,318 functions (12,460 vulnerable = 45.6%), C/C++ from QEMU/FFmpeg.

Usage:
    python reveal_pipeline.py --prepare   # create splits + obfuscation
    python reveal_pipeline.py --tfidf     # TF-IDF + LR eval (CPU, fast)
    python reveal_pipeline.py --prepare --tfidf
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

THESIS_ROOT  = Path(__file__).parent
REVEAL_SRC   = THESIS_ROOT / "data/raw/ReVeal/data/function.json"
DATASET_DIR  = THESIS_ROOT / "reveal_dataset"
SPLITS_DIR   = DATASET_DIR / "splits"
RESULTS_DIR  = THESIS_ROOT / "devign_full"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_source():
    with open(REVEAL_SRC) as f:
        rows = json.load(f)
    return [{"func": r["func"], "target": int(r["target"])} for r in rows]


def prepare_splits():
    train_file = SPLITS_DIR / "train.jsonl"
    if train_file.exists():
        print("Splits already exist — skipping split creation.")
    else:
        print(f"Loading {REVEAL_SRC}...")
        rows = load_source()
        print(f"  {len(rows)} functions, vul={sum(r['target'] for r in rows)}")

        # Stratified 80/10/10
        pos = [r for r in rows if r['target'] == 1]
        neg = [r for r in rows if r['target'] == 0]
        random.seed(42)
        random.shuffle(pos); random.shuffle(neg)

        def split3(lst):
            n = len(lst)
            n_tr = int(0.8 * n); n_va = int(0.1 * n)
            return lst[:n_tr], lst[n_tr:n_tr+n_va], lst[n_tr+n_va:]

        pos_tr, pos_va, pos_te = split3(pos)
        neg_tr, neg_va, neg_te = split3(neg)

        splits = {
            'train': pos_tr + neg_tr,
            'valid': pos_va + neg_va,
            'test':  pos_te + neg_te,
        }
        for name, data in splits.items():
            random.shuffle(data)
            out = SPLITS_DIR / f"{name}.jsonl"
            with open(out, 'w') as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")
            n_vul = sum(r['target'] for r in data)
            print(f"  {name}: {len(data)} (vul={n_vul}, clean={len(data)-n_vul})")

    # Apply obfuscation to test set
    _apply_obfuscation()


def _apply_obfuscation():
    test_file = SPLITS_DIR / "test.jsonl"
    if not test_file.exists():
        print("ERROR: test.jsonl missing — run --prepare first")
        return

    already_done = all(
        (SPLITS_DIR / f"test_obf_{k}.jsonl").exists()
        for k in ("identifier", "deadcode", "controlflow")
    )
    if already_done:
        print("Obfuscated test splits already exist — skipping.")
        return

    sys.path.insert(0, str(THESIS_ROOT / "devign_full"))
    try:
        from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow
    except ImportError as e:
        print(f"ERROR importing obf_transforms_v2: {e}")
        return

    with open(test_file) as f:
        test_rows = [json.loads(l) for l in f]

    transforms = {
        "test_obf_identifier":  obf_identifier,
        "test_obf_deadcode":    obf_deadcode,
        "test_obf_controlflow": obf_controlflow,
    }
    for name, fn in transforms.items():
        out_path = SPLITS_DIR / f"{name}.jsonl"
        if out_path.exists():
            print(f"  {name}: already exists, skipping")
            continue
        obf_rows = []; failed = 0
        for row in test_rows:
            try:
                obf_func = fn(row['func'])
                obf_rows.append({"func": obf_func, "target": row['target']})
            except Exception:
                obf_rows.append(row); failed += 1
        with open(out_path, 'w') as f:
            for r in obf_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(obf_rows)} samples ({failed} fallback to original)")


def run_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print("\n=== TF-IDF + LR on ReVeal ===\n")

    def load_jsonl(path):
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        return [r['func'] for r in rows], np.array([r['target'] for r in rows], dtype=int)

    train_texts, train_labels = load_jsonl(SPLITS_DIR / "train.jsonl")
    print(f"Train: {len(train_texts)} (vul={train_labels.sum()})")

    N_TRIALS  = 10
    SUBSAMPLE = 5000

    TFIDF_PARAMS = dict(
        max_features=10000, sublinear_tf=True, ngram_range=(1, 2),
        analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3,
    )
    CONDITIONS = {
        "original":    "test.jsonl",
        "identifier":  "test_obf_identifier.jsonl",
        "deadcode":    "test_obf_deadcode.jsonl",
        "controlflow": "test_obf_controlflow.jsonl",
    }
    test_sets = {}
    for cond, fname in CONDITIONS.items():
        p = SPLITS_DIR / fname
        if p.exists():
            test_sets[cond] = load_jsonl(p)
        else:
            print(f"  WARN: {p} not found — skipping {cond}")

    trial_results = {c: [] for c in test_sets}
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(42 + trial)
        idx = rng.choice(len(train_texts), size=min(SUBSAMPLE, len(train_texts)), replace=False)
        sub_texts  = [train_texts[i] for i in idx]
        sub_labels = train_labels[idx]

        vec = TfidfVectorizer(**TFIDF_PARAMS)
        X_tr = vec.fit_transform(sub_texts)
        clf  = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X_tr, sub_labels)

        for cond, (texts, labels) in test_sets.items():
            X_te = vec.transform(texts)
            preds = clf.predict(X_te)
            trial_results[cond].append(f1_score(labels, preds, zero_division=0))
        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{N_TRIALS} done")

    results = {}
    base_f1 = None
    for cond, f1s in trial_results.items():
        arr = np.array(f1s) * 100
        entry = {"f1_mean": round(float(arr.mean()), 2),
                 "f1_std":  round(float(arr.std()),  2),
                 "all_f1":  [round(x, 2) for x in arr.tolist()]}
        if cond == "original":
            base_f1 = entry["f1_mean"]
        else:
            entry["delta_f1"] = round(entry["f1_mean"] - base_f1, 2) if base_f1 else 0
        results[cond] = entry
        delta = f"  ΔF1={entry.get('delta_f1',0):+.2f}" if cond != "original" else ""
        print(f"  {cond:<20} F1={entry['f1_mean']:.2f}±{entry['f1_std']:.2f}%{delta}")

    out = RESULTS_DIR / "reveal_tfidf_results.json"
    with open(out, 'w') as f:
        json.dump({"dataset": "ReVeal", "model": "TF-IDF+LR",
                   "n_trials": N_TRIALS, "results": results}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tfidf",   action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare_splits()
    if args.tfidf:
        run_tfidf()
    if not args.prepare and not args.tfidf:
        parser.print_help()
