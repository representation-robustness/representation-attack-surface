#!/usr/bin/env python3
"""
CodeBERT fine-tuning for vulnerability detection on devign_full.

Uses microsoft/codebert-base, which was pre-trained on code + natural language
pairs across 6 programming languages including C/C++.

We fine-tune for binary sequence classification (vulnerable / not vulnerable)
using the raw C source code of each function as input.

Key difference from CPG+GNN approach:
  - Input is raw token sequence, not graph structure
  - BPE tokenization handles renamed identifiers better than Word2Vec
    (unknown tokens are split into sub-word pieces rather than becoming OOV)
  - No graph construction, no Joern, no edge types
  - Pre-trained on 6M code+doc pairs: rich semantic priors

Expected outcome from literature (CodeXGLUE benchmark):
  - CodeBERT on Devign: ~63-65% accuracy / F1
  - This should be non-degenerate (real discrimination, not all-positive)

Robustness evaluation:
  After training, evaluate on original test + 3 obfuscated test sets,
  reporting F1 and ΔF1 for each obfuscation type.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import copy

SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_ROOT  = THESIS_ROOT / "devign_full"
SPLIT_FILE   = DEVIGN_ROOT / "devign_full_split_801010.json"
MODEL_DIR    = SCRIPT_DIR / "models" / "codebert_devign"

MODEL_NAME   = "microsoft/codebert-base"
MAX_LENGTH   = 512
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
PATIENCE     = 3

DATA_FILES = {
    "originals":      DEVIGN_ROOT / "originals_full_data_with_slices.json",
    "obf_identifier": DEVIGN_ROOT / "obf_identifier_full_data_with_slices.json",
    "obf_deadcode":   DEVIGN_ROOT / "obf_deadcode_full_data_with_slices.json",
    "obf_controlflow":DEVIGN_ROOT / "obf_controlflow_full_data_with_slices.json",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=MAX_LENGTH):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        code = rec["code"]
        label = int(rec["label"])
        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


def make_loader(records, tokenizer, batch_size, balanced=False, shuffle=False):
    ds = CodeDataset(records, tokenizer)
    if balanced:
        labels = [int(r["label"]) for r in records]
        pos = sum(labels); neg = len(labels) - pos
        w = [1.0/neg if l == 0 else 1.0/pos for l in labels]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"]
        outputs   = model(input_ids=input_ids, attention_mask=attn_mask)
        preds     = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds) * 100
    pr  = precision_score(all_labels, all_preds, zero_division=0) * 100
    rc  = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1  = f1_score(all_labels, all_preds, zero_division=0) * 100
    return acc, pr, rc, f1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_loader, valid_loader, device, num_epochs, warmup_steps):
    total_steps = num_epochs * len(train_loader)
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Linear warmup then constant
    warmup_sched   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                               total_iters=warmup_steps)
    constant_sched = ConstantLR(optimizer, factor=1.0,
                                 total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, constant_sched],
                              milestones=[warmup_steps])

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    no_improve  = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            outputs   = model(input_ids=input_ids, attention_mask=attn_mask,
                              labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_steps    += 1

        val_acc, val_pr, val_rc, val_f1 = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch}/{num_epochs}  "
              f"loss={total_loss/n_steps:.4f}  "
              f"val_Acc={val_acc:.2f}%  val_Pr={val_pr:.2f}%  "
              f"val_Rc={val_rc:.2f}%  val_F1={val_f1:.2f}%", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
            print(f"  -> New best val F1={val_f1:.2f}%  "
                  f"Pr={val_pr:.2f}%  Rc={val_rc:.2f}%", flush=True)
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{PATIENCE})", flush=True)
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}  "
              f"{free/1e9:.1f}/{total/1e9:.1f} GB free", flush=True)

    # Load split
    print("\nLoading split and source data…", flush=True)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_files = set(split["splits"]["train"])
    valid_files = set(split["splits"]["valid"])
    test_files  = set(split["splits"]["test"])

    with open(DATA_FILES["originals"]) as f:
        orig = json.load(f)

    idx = {d["file_name"]: d for d in orig}
    train_recs = [idx[f] for f in train_files if f in idx]
    valid_recs = [idx[f] for f in valid_files if f in idx]
    test_recs  = [idx[f] for f in test_files  if f in idx]

    pos = sum(int(r["label"]) for r in train_recs)
    neg = len(train_recs) - pos
    print(f"  train={len(train_recs)} (pos={pos}, neg={neg})  "
          f"valid={len(valid_recs)}  test={len(test_recs)}", flush=True)

    # Tokenizer + model
    print(f"\nLoading {MODEL_NAME}…", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model     = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    # Data loaders
    train_loader = make_loader(train_recs, tokenizer, BATCH_SIZE, balanced=True)
    valid_loader = make_loader(valid_recs, tokenizer, BATCH_SIZE * 2)
    test_loader  = make_loader(test_recs,  tokenizer, BATCH_SIZE * 2)

    warmup_steps = int(len(train_loader) * NUM_EPOCHS * WARMUP_RATIO)
    print(f"\nFine-tuning: {NUM_EPOCHS} epochs  "
          f"batch={BATCH_SIZE}  lr={LR}  "
          f"warmup_steps={warmup_steps}", flush=True)

    model, best_val_f1 = train(
        model, train_loader, valid_loader, device, NUM_EPOCHS, warmup_steps)

    # Test set evaluation
    t_acc, t_pr, t_rc, t_f1 = evaluate(model, test_loader, device)
    print(f"\nOriginal test:  Acc={t_acc:.2f}%  Pr={t_pr:.2f}%  "
          f"Rc={t_rc:.2f}%  F1={t_f1:.2f}%", flush=True)

    degenerate = t_rc > 93.0 or t_f1 <= 63.7
    print(f"{'DEGENERATE' if degenerate else 'NON-DEGENERATE'}  "
          f"(Pr={t_pr:.1f}%, Rc={t_rc:.1f}%)", flush=True)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\nModel saved → {MODEL_DIR}", flush=True)

    # Robustness evaluation
    print("\n" + "="*60, flush=True)
    print("ROBUSTNESS EVALUATION", flush=True)
    print("="*60, flush=True)

    results = {"original": {"acc": t_acc, "pr": t_pr, "rc": t_rc, "f1": t_f1}}

    for obf_name, obf_file in [
        ("obf_identifier",  DATA_FILES["obf_identifier"]),
        ("obf_deadcode",    DATA_FILES["obf_deadcode"]),
        ("obf_controlflow", DATA_FILES["obf_controlflow"]),
    ]:
        with open(obf_file) as f:
            obf_data = json.load(f)
        obf_idx  = {d["file_name"]: d for d in obf_data}
        obf_recs = [obf_idx[f] for f in test_files if f in obf_idx]
        obf_loader = make_loader(obf_recs, tokenizer, BATCH_SIZE * 2)
        acc, pr, rc, f1 = evaluate(model, obf_loader, device)
        delta = f1 - t_f1
        results[obf_name] = {"acc": acc, "pr": pr, "rc": rc, "f1": f1,
                              "delta_f1": delta}
        print(f"  {obf_name:20s}  Acc={acc:.2f}%  Pr={pr:.2f}%  "
              f"Rc={rc:.2f}%  F1={f1:.2f}%  ΔF1={delta:+.2f}%", flush=True)

    print(f"\nRobustness summary (ΔF1 from baseline F1={t_f1:.2f}%):", flush=True)
    for k, v in results.items():
        if k != "original":
            print(f"  {k:20s}  ΔF1={v['delta_f1']:+.2f}%", flush=True)

    # Save results
    out = {
        "model":      MODEL_NAME,
        "best_val_f1": best_val_f1,
        "test_original": results["original"],
        "robustness": {k: v for k, v in results.items() if k != "original"},
    }
    with open(SCRIPT_DIR / "codebert_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {SCRIPT_DIR/'codebert_results.json'}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
