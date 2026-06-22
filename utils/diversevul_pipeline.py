"""
diversevul_pipeline.py — Build balanced DiverseVul Dataset 3 splits,
apply obfuscation transforms, and run TF-IDF+LR baseline.

Source: ~/thesis/diversevul/diversevul_all.jsonl (330,492 functions)
Strategy:
  - Filter QEMU + FFmpeg (overlap risk with Devign/BigVul)
  - Undersample clean to match vulnerable count (50/50 balance, ~36k total)
  - Stratified 80/10/10 split
  - Apply 3 obfuscation transforms to test set

Usage:
    python diversevul_pipeline.py --split   # build splits + obfuscation
    python diversevul_pipeline.py --tfidf   # TF-IDF+LR evaluation (CPU)
    python diversevul_pipeline.py --split --tfidf
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

THESIS_ROOT  = Path(__file__).parent
SOURCE_JSONL = THESIS_ROOT / "diversevul" / "diversevul_all.jsonl"
SPLITS_DIR   = THESIS_ROOT / "diversevul_dataset" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = THESIS_ROOT / "devign_full"
OBF_SCRIPT   = THESIS_ROOT / "devign_full" / "obf_transforms_v2.py"

# Mirror to ReGVD dataset dir
REGVD_DIR = Path.home() / "GNN-ReGVD" / "dataset_diversevul"
REGVD_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_PROJECTS = {"qemu", "ffmpeg"}
BALANCE_SEED = 42


def load_and_filter():
    print(f"Loading {SOURCE_JSONL} ...")
    rows = []
    with open(SOURCE_JSONL) as f:
        for line in f:
            r = json.loads(line)
            proj = r.get("project", "").lower()
            if proj not in EXCLUDE_PROJECTS:
                rows.append({"func": r["func"], "target": int(r["target"])})
    vuln  = [r for r in rows if r["target"] == 1]
    clean = [r for r in rows if r["target"] == 0]
    print(f"After filter: {len(rows)} total  |  vuln={len(vuln)}  clean={len(clean)}")
    return vuln, clean


def build_splits():
    train_file = SPLITS_DIR / "train.jsonl"
    if train_file.exists():
        print("Splits already exist — skipping split creation.")
        return

    vuln, clean = load_and_filter()

    rng = random.Random(BALANCE_SEED)
    n = len(vuln)
    sampled_clean = rng.sample(clean, n)
    all_rows = vuln + sampled_clean
    rng.shuffle(all_rows)
    print(f"Balanced dataset: {len(all_rows)} total  ({n} vuln + {n} clean)")

    n_total = len(all_rows)
    n_train = int(0.8 * n_total)
    n_valid = int(0.1 * n_total)
    splits = {
        "train": all_rows[:n_train],
        "valid": all_rows[n_train:n_train + n_valid],
        "test":  all_rows[n_train + n_valid:],
    }

    for name, data in splits.items():
        out = SPLITS_DIR / f"{name}.jsonl"
        with open(out, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        n_vul = sum(1 for r in data if r["target"] == 1)
        print(f"  {name}: {len(data)} samples  (vul={n_vul}  clean={len(data)-n_vul})")
        # Mirror to ReGVD dir
        regvd_out = REGVD_DIR / f"{name}.jsonl"
        regvd_out.write_text(out.read_text())

    print("\nApplying obfuscation to test set ...")
    _apply_obfuscation(splits["test"])


def _apply_obfuscation(test_rows=None):
    sys.path.insert(0, str(THESIS_ROOT / "devign_full"))
    try:
        from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow
    except ImportError:
        try:
            from obf_transforms import transform_identifiers as obf_identifier, \
                insert_deadcode as obf_deadcode, rewrite_controlflow as obf_controlflow
        except ImportError:
            print("WARN: obf_transforms not found — skipping obfuscation")
            return

    if test_rows is None:
        test_file = SPLITS_DIR / "test.jsonl"
        with open(test_file) as f:
            test_rows = [json.loads(l) for l in f]

    transforms = {
        "test_obf_identifier":  obf_identifier,
        "test_obf_deadcode":    obf_deadcode,
        "test_obf_controlflow": obf_controlflow,
    }
    for name, fn in transforms.items():
        out = SPLITS_DIR / f"{name}.jsonl"
        if out.exists():
            print(f"  {name}: already exists")
            continue
        obf_rows = []
        failed = 0
        for row in test_rows:
            try:
                obf_rows.append({"func": fn(row["func"]), "target": row["target"]})
            except Exception:
                obf_rows.append(row)
                failed += 1
        with open(out, "w") as f:
            for r in obf_rows:
                f.write(json.dumps(r) + "\n")
        # Mirror to ReGVD dir
        (REGVD_DIR / f"{name}.jsonl").write_text(out.read_text())
        print(f"  {name}: {len(obf_rows)} samples ({failed} fallback)")


def run_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    print("\n=== TF-IDF + LR on DiverseVul ===")
    log_lines = ["\n=== TF-IDF + LR on DiverseVul ===\n"]

    def load_jsonl(path):
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        return [r["func"] for r in rows], np.array([r["target"] for r in rows], dtype=int)

    train_texts, train_labels = load_jsonl(SPLITS_DIR / "train.jsonl")
    log_lines.append(f"Train: {len(train_texts)} (vul={train_labels.sum()})\n")
    print(f"Train: {len(train_texts)} (vul={train_labels.sum()})")

    test_conditions = {
        "original":    "test.jsonl",
        "identifier":  "test_obf_identifier.jsonl",
        "deadcode":    "test_obf_deadcode.jsonl",
        "controlflow": "test_obf_controlflow.jsonl",
    }

    N_TRIALS = 10
    results = {}
    for cond, fname in test_conditions.items():
        fpath = SPLITS_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {cond}: {fpath} not found")
            continue
        test_texts, test_labels = load_jsonl(fpath)
        all_f1 = []
        for trial in range(N_TRIALS):
            seed = 42 + trial
            rng  = np.random.RandomState(seed)
            idx  = rng.choice(len(train_texts), min(10000, len(train_texts)), replace=False)
            sub_texts  = [train_texts[i] for i in idx]
            sub_labels = train_labels[idx]
            vec = TfidfVectorizer(
                max_features=10000, sublinear_tf=True, ngram_range=(1, 2),
                analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3,
            )
            X_tr = vec.fit_transform(sub_texts)
            X_te = vec.transform(test_texts)
            clf  = LogisticRegression(class_weight="balanced", max_iter=1000,
                                      C=1.0, random_state=seed)
            clf.fit(X_tr, sub_labels)
            preds = clf.predict(X_te)
            all_f1.append(f1_score(test_labels, preds, zero_division=0) * 100)
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{N_TRIALS} done", flush=True)

        results[cond] = {
            "f1_mean": round(float(np.mean(all_f1)), 2),
            "f1_std":  round(float(np.std(all_f1)),  2),
            "all_f1":  [round(x, 2) for x in all_f1],
        }

    base = results.get("original", {}).get("f1_mean", 0)
    for cond in results:
        if cond != "original":
            results[cond]["delta_f1"] = round(results[cond]["f1_mean"] - base, 2)

    for cond, r in results.items():
        delta = f"  ΔF1={r.get('delta_f1', 0):+.2f}" if cond != "original" else ""
        line = f"  {cond:<20} F1={r['f1_mean']:.2f}±{r['f1_std']:.2f}%{delta}"
        print(line)
        log_lines.append(line + "\n")

    out_json = RESULTS_DIR / "diversevul_tfidf_results.json"
    with open(out_json, "w") as f:
        json.dump({"dataset": "DiverseVul", "model": "TF-IDF+LR",
                   "n_trials": N_TRIALS, "results": results}, f, indent=2)
    msg = f"\nSaved → {out_json}"
    print(msg)
    log_lines.append(msg + "\n")

    log_out = Path.home() / "diversevul_logs" / "tfidf.log"
    log_out.parent.mkdir(exist_ok=True)
    with open(log_out, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  action="store_true", help="Build balanced splits + obfuscation")
    parser.add_argument("--tfidf",  action="store_true", help="Run TF-IDF+LR baseline")
    args = parser.parse_args()

    if not args.split and not args.tfidf:
        parser.print_help()
        sys.exit(1)

    if args.split:
        build_splits()
    if args.tfidf:
        run_tfidf()
