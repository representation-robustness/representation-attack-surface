"""
bigvul_pipeline.py — Download Big-Vul dataset, preprocess, apply obfuscation,
and evaluate TF-IDF + standalone CodeBERT models as cross-dataset generalization check.

Big-Vul: Fan et al. MSR 2021, ~188k C/C++ functions with CVE labels.

Usage:
    python bigvul_pipeline.py --download     # download + preprocess
    python bigvul_pipeline.py --tfidf        # run TF-IDF eval (CPU, fast)
    python bigvul_pipeline.py --codebert     # fine-tune CodeBERT (GPU)
    python bigvul_pipeline.py --download --tfidf --codebert   # all steps
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

THESIS_ROOT = Path(__file__).parent
BIGVUL_DIR  = THESIS_ROOT / "bigvul"
BIGVUL_DIR.mkdir(exist_ok=True)

SPLITS_DIR  = BIGVUL_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

OBF_SCRIPT = THESIS_ROOT / "devign_full" / "obf_transforms_v2.py"
RESULTS_DIR = THESIS_ROOT / "devign_full"


# ── Download ──────────────────────────────────────────────────────────────────

def try_huggingface_download():
    """Try to download Big-Vul from HuggingFace datasets."""
    # Several community uploads of Big-Vul exist on HF
    candidates = [
        ("benjifisher/bigvul", None),
        ("VulnCodeBERT/big-vul", None),
        ("clm21/bigvul", None),
        ("msrvul/bigvul", None),
    ]
    from datasets import load_dataset
    for name, subset in candidates:
        try:
            print(f"  Trying HuggingFace: {name}...")
            ds = load_dataset(name, subset, trust_remote_code=True)
            print(f"  SUCCESS: {name}")
            return ds
        except Exception as e:
            print(f"  Failed ({type(e).__name__}): {e!s:.100}")
    return None


def try_github_download():
    """Try to wget Big-Vul CSV from GitHub or known mirrors."""
    import subprocess
    urls = [
        # MSR20 repo raw CSV (may be LFS-tracked, will get pointer not data)
        "https://raw.githubusercontent.com/ZeoVan/MSR_20_Code_vulnerability_data_KD/master/code.csv",
        # BigVul dataset hosted on HuggingFace Hub as a file
        "https://huggingface.co/datasets/benjifisher/bigvul/resolve/main/data/train-00000-of-00001.parquet",
    ]
    out = BIGVUL_DIR / "bigvul_raw.parquet"
    for url in urls:
        print(f"  Trying wget: {url}")
        r = subprocess.run(
            ["wget", "-q", "--timeout=60", "-O", str(out), url],
            capture_output=True
        )
        if r.returncode == 0 and out.stat().st_size > 100_000:
            print(f"  Downloaded {out.stat().st_size/1e6:.1f} MB")
            return out
        else:
            out.unlink(missing_ok=True)
            print(f"  Failed or too small")
    return None


def build_jsonl_from_hf(ds):
    """Convert HuggingFace dataset to our JSONL format (func + target)."""
    import pandas as pd
    # Try to figure out which split has the data
    split = 'train' if 'train' in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    print(f"  Columns: {list(df.columns)}")

    # Map column names to func/target
    func_col = next((c for c in ['func', 'func_after', 'code', 'function'] if c in df.columns), None)
    label_col = next((c for c in ['target', 'label', 'vul', 'vulnerable'] if c in df.columns), None)

    if func_col is None or label_col is None:
        print(f"  ERROR: Could not identify func/label columns in {list(df.columns)}")
        return None

    df = df[[func_col, label_col]].rename(columns={func_col: 'func', label_col: 'target'})
    df = df.dropna(subset=['func', 'target'])
    df['target'] = df['target'].astype(int)
    # Filter to C/C++ only and non-empty
    df = df[df['func'].str.len() > 20]
    print(f"  {len(df)} functions ({df.target.sum()} vulnerable, {(df.target==0).sum()} clean)")
    return df


def build_jsonl_from_parquet(parquet_path):
    """Convert parquet file to our JSONL format."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    print(f"  Columns: {list(df.columns)}")
    func_col  = next((c for c in ['func', 'func_after', 'code', 'function'] if c in df.columns), None)
    label_col = next((c for c in ['target', 'label', 'vul', 'vulnerable'] if c in df.columns), None)
    if func_col is None or label_col is None:
        print(f"  ERROR: cannot identify func/label in {list(df.columns)}")
        return None
    df = df[[func_col, label_col]].rename(columns={func_col: 'func', label_col: 'target'})
    df = df.dropna(subset=['func', 'target'])
    df['target'] = df['target'].astype(int)
    df = df[df['func'].str.len() > 20]
    print(f"  {len(df)} functions ({df.target.sum()} vul, {(df.target==0).sum()} clean)")
    return df


def download_and_preprocess():
    """Download Big-Vul and create train/valid/test JSONL splits."""
    all_jsonl = BIGVUL_DIR / "bigvul_all.jsonl"

    if all_jsonl.exists():
        print(f"Already downloaded: {all_jsonl}")
    else:
        print("Downloading Big-Vul dataset...")
        df = None

        # Try HuggingFace first
        try:
            ds = try_huggingface_download()
            if ds is not None:
                df = build_jsonl_from_hf(ds)
        except ImportError:
            print("  datasets library not available")

        # Try parquet from HF hub
        if df is None:
            pq = try_github_download()
            if pq is not None:
                df = build_jsonl_from_parquet(pq)

        if df is None:
            print("\nERROR: Could not download Big-Vul from any source.")
            print("Manual steps:")
            print("  1. Go to https://github.com/ZeoVan/MSR_20_Code_vulnerability_data_KD")
            print("  2. Download code.csv from the Google Drive link")
            print("  3. Place at ~/thesis/bigvul/bigvul_raw.csv")
            print("  Then re-run: python bigvul_pipeline.py --download")
            sys.exit(1)

        # Write JSONL
        with open(all_jsonl, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps({"func": row['func'], "target": int(row['target'])}) + "\n")
        print(f"Saved {len(df)} functions → {all_jsonl}")

    # Create 80/10/10 train/valid/test splits
    train_file = SPLITS_DIR / "train.jsonl"
    if train_file.exists():
        print("Splits already created.")
        return

    print("\nCreating 80/10/10 splits...")
    with open(all_jsonl) as f:
        rows = [json.loads(l) for l in f]

    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    splits = {
        'train': rows[:n_train],
        'valid': rows[n_train:n_train+n_valid],
        'test':  rows[n_train+n_valid:],
    }
    for name, data in splits.items():
        out = SPLITS_DIR / f"{name}.jsonl"
        with open(out, 'w') as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        n_vul = sum(1 for r in data if r['target'] == 1)
        print(f"  {name}: {len(data)} samples (vul={n_vul}, clean={len(data)-n_vul})")

    # Apply obfuscation to test set
    print("\nApplying obfuscation transforms to test set...")
    _apply_obfuscation()


def _apply_obfuscation():
    """Apply our 3 obfuscation transforms to the Big-Vul test set."""
    sys.path.insert(0, str(THESIS_ROOT / "devign_full"))
    try:
        from obf_transforms_v2 import (
            rename_identifiers, insert_dead_code, restructure_control_flow
        )
    except ImportError:
        try:
            from obf_transforms import (
                rename_identifiers, insert_dead_code, restructure_control_flow
            )
        except ImportError:
            print("  WARN: obf_transforms not found — skipping obfuscation")
            return

    test_file = SPLITS_DIR / "test.jsonl"
    with open(test_file) as f:
        test_rows = [json.loads(l) for l in f]

    transforms = {
        'test_obf_identifier':  rename_identifiers,
        'test_obf_deadcode':    insert_dead_code,
        'test_obf_controlflow': restructure_control_flow,
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
                obf_func = fn(row['func'])
                obf_rows.append({"func": obf_func, "target": row['target']})
            except Exception:
                obf_rows.append(row)
                failed += 1
        with open(out, 'w') as f:
            for r in obf_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {name}: {len(obf_rows)} samples ({failed} fallback to original)")


# ── TF-IDF evaluation ─────────────────────────────────────────────────────────

def run_tfidf():
    """Train TF-IDF+LR on Big-Vul train set, evaluate on 4 test conditions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print("\n=== TF-IDF + Logistic Regression on Big-Vul ===")

    def load_jsonl(path):
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        texts  = [r['func'] for r in rows]
        labels = np.array([r['target'] for r in rows], dtype=int)
        return texts, labels

    train_texts, train_labels = load_jsonl(SPLITS_DIR / "train.jsonl")
    print(f"Train: {len(train_texts)} samples")

    test_conditions = {
        "original":    "test.jsonl",
        "identifier":  "test_obf_identifier.jsonl",
        "deadcode":    "test_obf_deadcode.jsonl",
        "controlflow": "test_obf_controlflow.jsonl",
    }

    N_TRIALS = 5
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
            vectorizer = TfidfVectorizer(
                max_features=10000, sublinear_tf=True,
                ngram_range=(1,2), analyzer="word",
                token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3,
            )
            # Subsample 5k for speed
            idx = np.random.RandomState(seed).choice(len(train_texts), min(5000, len(train_texts)), replace=False)
            sub_texts  = [train_texts[i] for i in idx]
            sub_labels = train_labels[idx]
            X_tr = vectorizer.fit_transform(sub_texts)
            X_te = vectorizer.transform(test_texts)
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=seed)
            clf.fit(X_tr, sub_labels)
            preds = clf.predict(X_te)
            all_f1.append(f1_score(test_labels, preds, zero_division=0) * 100)

        results[cond] = {
            "f1_mean": round(float(np.mean(all_f1)), 2),
            "f1_std":  round(float(np.std(all_f1)), 2),
            "all_f1":  [round(x, 2) for x in all_f1],
        }
        print(f"  {cond:<20} F1={results[cond]['f1_mean']:.2f} ± {results[cond]['f1_std']:.2f}%")

    base = results.get("original", {}).get("f1_mean", 0)
    for cond in results:
        if cond != "original":
            results[cond]["delta_f1"] = round(results[cond]["f1_mean"] - base, 2)

    out = RESULTS_DIR / "bigvul_tfidf_results.json"
    with open(out, 'w') as f:
        json.dump({"dataset": "Big-Vul", "model": "TF-IDF+LR", "n_trials": N_TRIALS, "results": results}, f, indent=2)
    print(f"\nSaved → {out}")


# ── CodeBERT evaluation ───────────────────────────────────────────────────────

def run_codebert():
    """Fine-tune CodeBERT on Big-Vul and evaluate on 4 conditions."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print("\n=== CodeBERT fine-tune on Big-Vul ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    MAX_LEN = 512
    BATCH   = 16
    EPOCHS  = 5
    LR      = 2e-5
    CKPT    = str(BIGVUL_DIR / "codebert_bigvul.pt")

    class FuncDataset(Dataset):
        def __init__(self, path):
            with open(path) as f:
                rows = [json.loads(l) for l in f]
            self.funcs  = [r['func'][:3000] for r in rows]
            self.labels = [r['target'] for r in rows]
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            enc = tokenizer(self.funcs[i], max_length=MAX_LEN, padding='max_length',
                            truncation=True, return_tensors='pt')
            return (enc['input_ids'].squeeze(), enc['attention_mask'].squeeze(),
                    torch.tensor(self.labels[i], dtype=torch.long))

    print("Loading train data...")
    train_ds = FuncDataset(SPLITS_DIR / "train.jsonl")
    valid_ds = FuncDataset(SPLITS_DIR / "valid.jsonl")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Class weights
    labels = train_ds.labels
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    w = torch.tensor([len(labels)/(2*n_neg), len(labels)/(2*n_pos)], dtype=torch.float).to(device)

    import torch.nn.functional as F
    best_val_f1, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for ids, mask, labs in train_loader:
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = model(ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits, labs, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        # Validate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for ids, mask, labs in valid_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids, attention_mask=mask).logits
                y_pred.extend(logits.argmax(-1).cpu().tolist())
                y_true.extend(labs.tolist())
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  Epoch {epoch}/{EPOCHS} loss={total_loss/len(train_loader):.4f} val_F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT)
    print(f"  Best val F1={best_val_f1:.4f} at epoch {best_epoch}")

    # Evaluate on 4 conditions
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    test_conditions = {
        "original":    "test.jsonl",
        "identifier":  "test_obf_identifier.jsonl",
        "deadcode":    "test_obf_deadcode.jsonl",
        "controlflow": "test_obf_controlflow.jsonl",
    }
    results = {}
    for cond, fname in test_conditions.items():
        fpath = SPLITS_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {cond}")
            continue
        test_ds     = FuncDataset(fpath)
        test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)
        y_true, y_pred = [], []
        with torch.no_grad():
            for ids, mask, labs in test_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids, attention_mask=mask).logits
                y_pred.extend(logits.argmax(-1).cpu().tolist())
                y_true.extend(labs.tolist())
        f1  = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        pr  = precision_score(y_true, y_pred, zero_division=0)
        rc  = recall_score(y_true, y_pred, zero_division=0)
        results[cond] = {"f1": round(f1*100,2), "acc": round(acc*100,2),
                         "pr": round(pr*100,2), "rc": round(rc*100,2)}
        print(f"  {cond:<20} F1={f1*100:.2f}% Pr={pr*100:.2f}% Rc={rc*100:.2f}%")

    base = results.get("original", {}).get("f1", 0)
    for cond in results:
        if cond != "original":
            results[cond]["delta_f1"] = round(results[cond]["f1"] - base, 2)

    out = RESULTS_DIR / "bigvul_codebert_results.json"
    with open(out, 'w') as f:
        json.dump({"dataset": "Big-Vul", "model": "CodeBERT", "best_val_f1": round(best_val_f1*100,2),
                   "results": results}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",  action="store_true")
    parser.add_argument("--tfidf",     action="store_true")
    parser.add_argument("--codebert",  action="store_true")
    args = parser.parse_args()

    if not any([args.download, args.tfidf, args.codebert]):
        parser.print_help()
        sys.exit(1)

    if args.download:
        download_and_preprocess()
    if args.tfidf:
        run_tfidf()
    if args.codebert:
        run_codebert()
