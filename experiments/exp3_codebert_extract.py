#!/usr/bin/env python3
"""
Exp 3: Extract per-function CodeBERT predictions for all 7 test conditions.
Uses available checkpoints (seed42 from ckpts_multiseed; reruns other seeds).

Output: devign_full/attack/preds/codebert_preds.json
"""

import copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score

THESIS     = Path(__file__).resolve().parents[1]
DEVIGN     = THESIS / "devign_full"
CKPT_DIR   = THESIS / "baselines" / "codebert" / "ckpts_multiseed"
OUT_DIR    = DEVIGN / "attack" / "preds"
OUT_FILE   = OUT_DIR / "codebert_preds.json"
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 32

CONDITIONS = {
    "clean":    DEVIGN / "originals_full_data_with_slices.json",
    "ren":      DEVIGN / "obf_identifier_full_data_with_slices.json",
    "dead":     DEVIGN / "obf_deadcode_full_data_with_slices.json",
    "cf":       DEVIGN / "obf_controlflow_full_data_with_slices.json",
    "ren_dead": DEVIGN / "obf_ren_dead_full_data_with_slices.json",
    "ren_cf":   DEVIGN / "obf_ren_cf_full_data_with_slices.json",
    "dead_cf":  DEVIGN / "obf_dead_cf_full_data_with_slices.json",
    "compound": DEVIGN / "obf_compound_full_data_with_slices.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        enc = self.tokenizer(r["code"], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(r["label"])

def get_test_records(data_path, test_files):
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    return [idx[f] for f in test_files if f in idx]

def run_inference(model, records, tokenizer):
    model.eval()
    ds     = CodeDataset(records, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch, lbl in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            p     = out.logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            labels.extend(lbl.tolist())
    return preds, labels

# Load split
split      = json.loads(SPLIT_FILE.read_text())
split      = split.get("splits", split)
test_files = split["test"]

# Load true labels once
clean_recs  = get_test_records(CONDITIONS["clean"], test_files)
true_labels = [r["label"] for r in clean_recs]
N           = len(true_labels)
print(f"Test set size: {N}", flush=True)

# Load tokenizer once
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Checkpoints available
available_ckpts = {
    42: CKPT_DIR / "codebert_seed42.pt",
}
# Check if other seeds exist
for s in [1337, 7, 100, 999]:
    p = CKPT_DIR / f"codebert_seed{s}.pt"
    if p.exists():
        available_ckpts[s] = p

print(f"Available checkpoints: {list(available_ckpts.keys())}", flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

all_seeds = {}
for seed, ckpt_path in available_ckpts.items():
    print(f"\n── Seed {seed} ──", flush=True)
    set_seed(seed)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)

    seed_preds = {}
    for cond, data_path in CONDITIONS.items():
        if not Path(data_path).exists():
            print(f"  {cond}: data file not found, skipping", flush=True)
            continue
        recs = get_test_records(data_path, test_files)
        if len(recs) == 0:
            print(f"  {cond}: 0 records found, skipping", flush=True)
            continue
        # Align to true_labels length
        n    = min(N, len(recs))
        preds, _ = run_inference(model, recs[:n], tokenizer)
        if len(preds) < N:
            preds = preds + [0] * (N - len(preds))
        f1 = f1_score(true_labels[:n], preds[:n], zero_division=0) * 100
        print(f"  {cond}: F1={f1:.2f}%", flush=True)
        seed_preds[cond] = preds[:N]

    all_seeds[str(seed)] = seed_preds

out = {
    "model":       "codebert",
    "conditions":  list(CONDITIONS.keys()),
    "true_labels": true_labels,
    "seeds":       all_seeds,
}
OUT_FILE.write_text(json.dumps(out))
print(f"\nSaved → {OUT_FILE}", flush=True)
