#!/usr/bin/env python3
"""
Exp 14: Append vocab_shift, benign_token, and tempvar condition predictions
to existing text-model pred files (TF-IDF+LR, CodeT5+, CodeBERT, ReGVD).

For each model, loads existing preds/xxx_preds.json, runs inference only on
the three new conditions (skips any already present), and saves in-place.

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp14_new_attack_extract_text.py
"""

import json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THESIS     = Path(__file__).resolve().parents[1]
DEVIGN     = THESIS / "devign_full"
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"
PREDS_DIR  = DEVIGN / "attack" / "preds"

# New condition paths (written by exp13_new_attack_transforms.py + exp13b)
NEW_CONDITIONS = {
    "vocab_shift":  DEVIGN / "obf_vocab_shift_full_data_with_slices.json",
    "benign_token": DEVIGN / "obf_benign_token_full_data_with_slices.json",
    "tempvar":      DEVIGN / "obf_tempvar_full_data_with_slices.json",
    "datadep":      DEVIGN / "obf_datadep_full_data_with_slices.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def load_split():
    sp = json.loads(SPLIT_FILE.read_text())
    return sp["splits"]["test"], sp["splits"]["train"]


def get_test_recs(data_path, test_files):
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    return [idx[f] for f in test_files if f in idx]


def missing_conditions(preds, test_files):
    """Return dict of conditions not yet present for all seeds."""
    needed = {}
    for name, path in NEW_CONDITIONS.items():
        if not path.exists():
            print(f"  WARN: {path.name} not found, skipping {name}")
            continue
        # Check if already in all seeds
        all_seeds_have = all(name in preds["seeds"].get(sk, {}) for sk in preds["seeds"])
        if not all_seeds_have:
            needed[name] = path
    return needed


# ---------------------------------------------------------------------------
# TF-IDF + LR
# ---------------------------------------------------------------------------

def run_tfidf(test_files, train_files):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    pred_file = PREDS_DIR / "tfidf_preds.json"
    if not pred_file.exists():
        print("  SKIP tfidf: preds file not found"); return

    preds = json.loads(pred_file.read_text())
    needed = missing_conditions(preds, test_files)
    if not needed:
        print("  tfidf: all new conditions already present"); return

    # Build vectorizer + classifier on clean train data
    clean_map = {r["file_name"]: r for r in
                 json.loads((DEVIGN / "originals_full_data_with_slices.json").read_text())}
    train_texts  = [clean_map[f]["code"] for f in train_files if f in clean_map]
    train_labels = [int(clean_map[f]["label"]) for f in train_files if f in clean_map]

    TFIDF_PARAMS = dict(max_features=10000, sublinear_tf=True, ngram_range=(1, 2),
                        analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3)
    N_TRIALS = 5
    print(f"  tfidf: adding {list(needed.keys())}", flush=True)

    # tfidf_preds.json uses a single "42" seed key (majority vote of N trials stored there)
    seed_key = list(preds["seeds"].keys())[0]
    seed_preds = preds["seeds"][seed_key]

    for cond_name, cond_path in needed.items():
        test_recs  = get_test_recs(cond_path, test_files)
        test_texts = [r["code"] for r in test_recs]
        trial_preds = []
        for t in range(N_TRIALS):
            set_seed(42 + t * 13)
            vec = TfidfVectorizer(**TFIDF_PARAMS)
            X_tr = vec.fit_transform(train_texts)
            X_te = vec.transform(test_texts)
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42+t*13)
            clf.fit(X_tr, train_labels)
            trial_preds.append(clf.predict(X_te).tolist())
        majority = (np.array(trial_preds).mean(axis=0) >= 0.5).astype(int).tolist()
        seed_preds[cond_name] = majority
        print(f"    {cond_name}: done ({len(majority)} preds)", flush=True)

    if "conditions" in preds:
        for c in needed:
            if c not in preds["conditions"]:
                preds["conditions"].append(c)
    pred_file.write_text(json.dumps(preds, indent=2))
    print(f"  tfidf: saved → {pred_file}", flush=True)


# ---------------------------------------------------------------------------
# CodeT5+
# ---------------------------------------------------------------------------

def run_codet5plus(test_files):
    from transformers import T5EncoderModel, RobertaTokenizer

    pred_file = PREDS_DIR / "codet5plus_preds.json"
    if not pred_file.exists():
        print("  SKIP codet5+: preds file not found"); return

    preds = json.loads(pred_file.read_text())
    needed = missing_conditions(preds, test_files)
    if not needed:
        print("  codet5+: all new conditions already present"); return

    CKPT_DIR   = THESIS / "baselines" / "codebert" / "ckpts_codet5plus"
    MODEL_NAME = "Salesforce/codet5p-220m"
    MAX_LEN    = 512
    BATCH_SIZE = 32
    SEEDS      = [int(s) for s in preds["seeds"].keys()]

    class CodeT5PlusClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = T5EncoderModel.from_pretrained(MODEL_NAME)
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 2))
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

    print(f"  codet5+: loading tokenizer ...", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def batch_infer(model, texts):
        preds_out = []
        model.eval()
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad():
                logits = model(ids, mask)
            preds_out.extend(logits.argmax(-1).cpu().tolist())
        return preds_out

    print(f"  codet5+: adding {list(needed.keys())} for seeds {SEEDS}", flush=True)

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"codet5plus_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  seed {seed}: checkpoint missing, skip", flush=True); continue

        model = CodeT5PlusClassifier().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)

        seed_key = str(seed)
        for cond_name, cond_path in needed.items():
            if cond_name in preds["seeds"].get(seed_key, {}):
                continue
            test_recs  = get_test_recs(cond_path, test_files)
            test_texts = [r["code"] for r in test_recs]
            result = batch_infer(model, test_texts)
            preds["seeds"].setdefault(seed_key, {})[cond_name] = result
            print(f"    seed={seed} {cond_name}: done ({len(result)})", flush=True)

        del model
        torch.cuda.empty_cache()

    if "conditions" in preds:
        for c in needed:
            if c not in preds["conditions"]:
                preds["conditions"].append(c)
    pred_file.write_text(json.dumps(preds, indent=2))
    print(f"  codet5+: saved → {pred_file}", flush=True)


# ---------------------------------------------------------------------------
# CodeBERT
# ---------------------------------------------------------------------------

def run_codebert(test_files):
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    pred_file = PREDS_DIR / "codebert_preds.json"
    if not pred_file.exists():
        print("  SKIP codebert: preds file not found"); return

    preds = json.loads(pred_file.read_text())
    needed = missing_conditions(preds, test_files)
    if not needed:
        print("  codebert: all new conditions already present"); return

    CKPT_DIR   = THESIS / "baselines" / "codebert" / "ckpts_multiseed"
    MODEL_NAME = "microsoft/codebert-base"
    MAX_LEN    = 512
    BATCH_SIZE = 32
    SEEDS      = [int(s) for s in preds["seeds"].keys()]

    print(f"  codebert: loading tokenizer ...", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def batch_infer(model, recs):
        out = []
        model.eval()
        for i in range(0, len(recs), BATCH_SIZE):
            batch = recs[i:i+BATCH_SIZE]
            enc = tokenizer([r["code"] for r in batch], truncation=True,
                            max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            out.extend(logits.argmax(-1).cpu().tolist())
        return out

    print(f"  codebert: adding {list(needed.keys())} for seeds {SEEDS}", flush=True)

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"codebert_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  seed {seed}: checkpoint missing, skip", flush=True); continue

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # state may be full checkpoint dict or just state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

        seed_key = str(seed)
        for cond_name, cond_path in needed.items():
            if cond_name in preds["seeds"].get(seed_key, {}):
                continue
            test_recs = get_test_recs(cond_path, test_files)
            result    = batch_infer(model, test_recs)
            preds["seeds"].setdefault(seed_key, {})[cond_name] = result
            print(f"    seed={seed} {cond_name}: done ({len(result)})", flush=True)

        del model
        torch.cuda.empty_cache()

    if "conditions" in preds:
        for c in needed:
            if c not in preds["conditions"]:
                preds["conditions"].append(c)
    pred_file.write_text(json.dumps(preds, indent=2))
    print(f"  codebert: saved → {pred_file}", flush=True)


# ---------------------------------------------------------------------------
# ReGVD
# ---------------------------------------------------------------------------

def run_regvd(test_files):
    REGVD_DIR = THESIS / "baselines" / "regvd"
    sys.path.insert(0, str(REGVD_DIR))
    from train_regvd import ReGVDDataset, ReGVDModel, CODEBERT_MODEL, HIDDEN_DIM, NUM_GNN_LAYERS
    from torch_geometric.loader import DataLoader as PyGLoader

    pred_file = PREDS_DIR / "regvd_preds.json"
    if not pred_file.exists():
        print("  SKIP regvd: preds file not found"); return

    preds = json.loads(pred_file.read_text())
    needed = missing_conditions(preds, test_files)
    if not needed:
        print("  regvd: all new conditions already present"); return

    ckpt = THESIS / "baselines" / "regvd" / "models" / "regvd_devign" / "best.pt"
    if not ckpt.exists():
        print("  regvd: checkpoint not found, skip"); return

    print(f"  regvd: adding {list(needed.keys())}", flush=True)

    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    # Extract embedding weight matrix (frozen; same as training)
    cb_tmp = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = cb_tmp.embeddings.word_embeddings.weight.detach().cpu()
    del cb_tmp
    print(f"  regvd: embed_weight shape {embed_weight.shape}", flush=True)

    model = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    seed_key = list(preds["seeds"].keys())[0]
    BATCH_SIZE = 32

    for cond_name, cond_path in needed.items():
        test_recs = get_test_recs(cond_path, test_files)
        dataset   = ReGVDDataset(test_recs, embed_weight, tokenizer)
        loader    = PyGLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                all_preds.extend(out.argmax(-1).cpu().tolist())
        preds["seeds"][seed_key][cond_name] = all_preds
        print(f"    {cond_name}: done ({len(all_preds)})", flush=True)

    if "conditions" in preds:
        for c in needed:
            if c not in preds["conditions"]:
                preds["conditions"].append(c)
    pred_file.write_text(json.dumps(preds, indent=2))
    print(f"  regvd: saved → {pred_file}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_files, train_files = load_split()
    print(f"Test: {len(test_files)}  Train: {len(train_files)}", flush=True)

    missing = [k for k, p in NEW_CONDITIONS.items() if not p.exists()]
    if missing:
        print(f"Missing transform files: {missing}")
        print("Run exp13_new_attack_transforms.py first."); raise SystemExit(1)

    print("\n=== TF-IDF+LR ===", flush=True)
    run_tfidf(test_files, train_files)

    print("\n=== CodeBERT ===", flush=True)
    run_codebert(test_files)

    print("\n=== CodeT5+ ===", flush=True)
    run_codet5plus(test_files)

    print("\n=== ReGVD ===", flush=True)
    run_regvd(test_files)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
