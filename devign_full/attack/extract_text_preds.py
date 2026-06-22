#!/usr/bin/env python3
"""
Extract per-function predictions for TF-IDF+LR and CodeT5+ on all 7 conditions.
Saves preds/tfidf_preds.json and preds/codet5plus_preds.json.
"""

import json, os, sys, random
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THESIS_ROOT = Path(__file__).resolve().parents[2]
DEVIGN_FULL = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_FULL / "devign_full_split_801010.json"
OUT_DIR     = Path(__file__).parent / "preds"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "clean":    DEVIGN_FULL / "originals_full_data_with_slices.json",
    "ren":      DEVIGN_FULL / "obf_identifier_full_data_with_slices.json",
    "dead":     DEVIGN_FULL / "obf_deadcode_full_data_with_slices.json",
    "cf":       DEVIGN_FULL / "obf_controlflow_full_data_with_slices.json",
    "ren_dead": DEVIGN_FULL / "obf_ren_dead_full_data_with_slices.json",
    "ren_cf":   DEVIGN_FULL / "obf_ren_cf_full_data_with_slices.json",
    "dead_cf":  DEVIGN_FULL / "obf_dead_cf_full_data_with_slices.json",
    "compound": DEVIGN_FULL / "obf_compound_full_data_with_slices.json",
}

TFIDF_PARAMS = dict(max_features=10000, sublinear_tf=True, ngram_range=(1, 2),
                    analyzer="word", token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3)
N_TRIALS = 5
SUBSAMPLE = 5000


def load_text_splits():
    split = json.load(open(SPLIT_FILE))
    test_names = split["splits"]["test"]
    train_names = split["splits"]["train"]

    # load all conditions
    cond_data = {}
    for cond, path in CONDITIONS.items():
        recs = json.load(open(path))
        fn_map = {r["file_name"]: r for r in recs}
        cond_data[cond] = fn_map

    # build ordered test texts + true labels
    true_labels, test_texts = [], {}
    orig_map = cond_data["clean"]
    for n in test_names:
        if n in orig_map:
            true_labels.append(int(orig_map[n]["label"]))

    for cond, fn_map in cond_data.items():
        texts = []
        for n in test_names:
            if n in fn_map:
                texts.append(fn_map[n]["code"])
        test_texts[cond] = texts

    # train texts
    train_texts, train_labels = [], []
    for n in train_names:
        if n in orig_map:
            train_texts.append(orig_map[n]["code"])
            train_labels.append(int(orig_map[n]["label"]))

    return train_texts, np.array(train_labels), test_texts, true_labels


def extract_tfidf():
    print("\n=== TF-IDF+LR ===", flush=True)
    train_texts, y_train, test_texts, true_labels = load_text_splits()
    print(f"  train={len(y_train)}, test={len(true_labels)}", flush=True)

    vec     = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vec.fit_transform(train_texts)
    X_test  = {cond: vec.transform(txts) for cond, txts in test_texts.items()}
    print(f"  Vocab: {len(vec.vocabulary_)}", flush=True)

    result = {"model": "tfidf", "conditions": list(CONDITIONS.keys()),
              "true_labels": true_labels, "seeds": {}}

    rng = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    for trial in range(N_TRIALS):
        seed = 42 + trial * 7
        rng2 = np.random.default_rng(seed)
        n_pos = int(SUBSAMPLE * len(pos_idx) / len(y_train))
        n_neg = SUBSAMPLE - n_pos
        idx = np.concatenate([
            rng2.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng2.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ])
        rng2.shuffle(idx)

        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1000, random_state=seed)
        clf.fit(X_train[idx], y_train[idx])

        seed_preds = {}
        for cond, X in X_test.items():
            preds = clf.predict(X).tolist()
            seed_preds[cond] = preds
            print(f"  Trial {trial+1} {cond}: F1={f1_score(true_labels, preds, zero_division=0)*100:.2f}%",
                  flush=True)
        result["seeds"][str(seed)] = seed_preds

    out = OUT_DIR / "tfidf_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


def extract_codet5plus():
    print("\n=== CodeT5+ ===", flush=True)
    from transformers import T5EncoderModel, RobertaTokenizer
    import torch.nn as nn

    CKPT_DIR   = THESIS_ROOT / "baselines" / "codebert" / "ckpts_codet5plus"
    MODEL_NAME = "Salesforce/codet5p-220m"
    MAX_LEN    = 512
    BATCH_SIZE = 32
    SEEDS      = [42, 1337, 7, 100, 999]

    class CodeT5PlusClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = T5EncoderModel.from_pretrained(MODEL_NAME)
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 2))
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

    split  = json.load(open(SPLIT_FILE))
    test_names = split["splits"]["test"]

    print("  Loading test data for all conditions...", flush=True)
    cond_texts, true_labels = {}, []
    for cond, path in CONDITIONS.items():
        recs = json.load(open(path))
        fn_map = {r["file_name"]: r for r in recs}
        texts = []
        for n in test_names:
            if n in fn_map:
                texts.append(fn_map[n]["code"])
                if cond == "clean":
                    true_labels.append(int(fn_map[n]["label"]))
        cond_texts[cond] = texts
    print(f"  test={len(true_labels)}", flush=True)

    print(f"  Loading tokenizer {MODEL_NAME}...", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    result = {"model": "codet5plus", "conditions": list(CONDITIONS.keys()),
              "true_labels": true_labels, "seeds": {}}

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"codet5plus_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  Seed {seed}: missing, skip", flush=True); continue

        model = CodeT5PlusClassifier().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        print(f"  Seed {seed}", flush=True)

        seed_preds = {}
        for cond, texts in cond_texts.items():
            all_preds = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i+BATCH_SIZE]
                enc = tokenizer(batch_texts, max_length=MAX_LEN, padding="max_length",
                                truncation=True, return_tensors="pt")
                with torch.no_grad():
                    logits = model(enc["input_ids"].to(DEVICE),
                                   enc["attention_mask"].to(DEVICE))
                all_preds.extend(logits.argmax(-1).cpu().tolist())
            seed_preds[cond] = all_preds
            print(f"    {cond}: F1={f1_score(true_labels, all_preds, zero_division=0)*100:.2f}%",
                  flush=True)
        result["seeds"][str(seed)] = seed_preds
        del model; torch.cuda.empty_cache()

    out = OUT_DIR / "codet5plus_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["tfidf", "codet5plus", "all"])
    args = ap.parse_args()
    print(f"Device: {DEVICE}", flush=True)
    if args.model in ("tfidf", "all"):     extract_tfidf()
    if args.model in ("codet5plus", "all"): extract_codet5plus()
