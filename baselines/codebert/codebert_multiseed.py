#!/usr/bin/env python3
"""
CodeBERT multiseed evaluation for vulnerability detection on Devign.

Runs 5 seeds to estimate natural variance and confirm whether obfuscation
deltas (ΔRen=-1.34, ΔDead=-0.97, ΔCF=-1.70) exceed the noise floor.

Outputs:
    ~/thesis/devign_full/codebert_multiseed_results.json
"""

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
DEVIGN_ROOT = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_ROOT / "devign_full_split_801010.json"
CKPT_DIR    = SCRIPT_DIR / "ckpts_multiseed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "microsoft/codebert-base"
MAX_LENGTH   = 512
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
PATIENCE     = 3

SEEDS = [42, 1337, 7, 100, 999]

DATA_FILES = {
    "originals":       DEVIGN_ROOT / "originals_full_data_with_slices.json",
    "obf_identifier":  DEVIGN_ROOT / "obf_identifier_full_data_with_slices.json",
    "obf_deadcode":    DEVIGN_ROOT / "obf_deadcode_full_data_with_slices.json",
    "obf_controlflow": DEVIGN_ROOT / "obf_controlflow_full_data_with_slices.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["code"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(rec["label"]), dtype=torch.long),
        }


def make_loader(records, tokenizer, batch_size, balanced=False):
    ds = CodeDataset(records, tokenizer)
    if balanced:
        labels = [int(r["label"]) for r in records]
        pos = sum(labels); neg = len(labels) - pos
        w = [1.0/neg if l == 0 else 1.0/pos for l in labels]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        logits    = model(input_ids=input_ids, attention_mask=attn_mask).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        truths.extend(batch["labels"].tolist())
    return (f1_score(truths, preds, zero_division=0) * 100,
            accuracy_score(truths, preds) * 100,
            precision_score(truths, preds, zero_division=0) * 100,
            recall_score(truths, preds, zero_division=0) * 100)


def train_one_seed(seed, tokenizer, train_recs, valid_recs, test_splits):
    set_seed(seed)
    print(f"\n{'='*55}\n  CodeBERT Devign  Seed {seed}\n{'='*55}", flush=True)

    train_loader = make_loader(train_recs, tokenizer, BATCH_SIZE, balanced=True)
    valid_loader = make_loader(valid_recs, tokenizer, BATCH_SIZE * 2)
    test_loaders = {k: make_loader(v, tokenizer, BATCH_SIZE * 2)
                    for k, v in test_splits.items()}

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)

    total_steps  = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler    = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
         ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)],
        milestones=[warmup_steps])

    best_val_f1 = 0.0
    best_state  = copy.deepcopy(model.state_dict())
    no_improve  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = 0; n_steps = 0
        for batch in train_loader:
            iids = batch["input_ids"].to(DEVICE)
            amsk = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            loss = model(input_ids=iids, attention_mask=amsk, labels=lbls).loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); n_steps += 1

        val_f1, val_acc, _, _ = evaluate(model, valid_loader)
        print(f"  Ep {epoch}/{NUM_EPOCHS} loss={total_loss/n_steps:.4f} "
              f"val_F1={val_f1:.2f}%", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    results = {'seed': seed, 'best_val_f1': round(best_val_f1, 2)}

    for split_name, loader in test_loaders.items():
        f1, acc, pr, rc = evaluate(model, loader)
        results[split_name] = {'f1': round(f1, 2), 'acc': round(acc, 2),
                                'pr': round(pr, 2), 'rc': round(rc, 2)}
        print(f"  {split_name}: F1={f1:.2f}%", flush=True)

    del model; torch.cuda.empty_cache()
    return results


def aggregate(all_results):
    splits = ['test', 'test_obf_identifier', 'test_obf_deadcode', 'test_obf_controlflow']
    agg = {}
    for s in splits:
        f1s = [r[s]['f1'] for r in all_results if s in r]
        agg[s] = {'f1_mean': round(float(np.mean(f1s)), 2),
                  'f1_std':  round(float(np.std(f1s)), 2),
                  'all_f1':  f1s}
    base = agg['test']['f1_mean']
    for s in splits[1:]:
        agg[s]['delta_f1'] = round(agg[s]['f1_mean'] - base, 2)
    agg.update({'n_seeds': len(all_results), 'seeds': [r['seed'] for r in all_results],
                'model': 'CodeBERT', 'dataset': 'Devign'})
    return agg


def main():
    print(f"Device: {DEVICE}", flush=True)

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_names = set(split["splits"]["train"])
    valid_names = set(split["splits"]["valid"])
    test_names  = set(split["splits"]["test"])

    with open(DATA_FILES["originals"]) as f:
        orig = json.load(f)
    idx = {d["file_name"]: d for d in orig}

    train_recs = [idx[n] for n in split["splits"]["train"] if n in idx]
    valid_recs = [idx[n] for n in split["splits"]["valid"] if n in idx]
    test_recs  = [idx[n] for n in split["splits"]["test"]  if n in idx]

    test_splits = {"test": test_recs}
    for obf_key, obf_file in [
        ("test_obf_identifier",  DATA_FILES["obf_identifier"]),
        ("test_obf_deadcode",    DATA_FILES["obf_deadcode"]),
        ("test_obf_controlflow", DATA_FILES["obf_controlflow"]),
    ]:
        with open(obf_file) as f:
            obf_data = json.load(f)
        obf_idx  = {d["file_name"]: d for d in obf_data}
        test_splits[obf_key] = [obf_idx[n] for n in split["splits"]["test"] if n in obf_idx]

    print(f"train={len(train_recs)} valid={len(valid_recs)} test={len(test_recs)}", flush=True)
    print(f"Loading tokenizer {MODEL_NAME}…", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, tokenizer, train_recs, valid_recs, test_splits)
        all_results.append(r)
        per_seed_out = DEVIGN_ROOT / f"codebert_seed{seed}_results.json"
        with open(per_seed_out, 'w') as f:
            json.dump(r, f, indent=2)

    agg = aggregate(all_results)
    out = DEVIGN_ROOT / "codebert_multiseed_results.json"
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results → {out}", flush=True)

    base = agg['test']['f1_mean']
    print(f"\n  test:               F1={agg['test']['f1_mean']:.2f}±{agg['test']['f1_std']:.2f}%")
    for k in ['test_obf_identifier', 'test_obf_deadcode', 'test_obf_controlflow']:
        d = agg[k]
        print(f"  {k}: F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}% Δ={d['delta_f1']:+.2f}pp")


if __name__ == "__main__":
    main()
