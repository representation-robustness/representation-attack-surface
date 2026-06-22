#!/usr/bin/env python3
"""
Exp 8: CodeBERT mixed training (50% clean + 50% augmented, sampled each epoch).
Run 5 instances in parallel, one per GPU.

Usage:
    CUDA_VISIBLE_DEVICES=X python exp8_codebert_mixed_train.py --seed SEED

Output:
    baselines/codebert/ckpts_mixed/codebert_mixed_seedSEED.pt
    devign_full/codebert_mixed_seedSEED_results.json
"""

import argparse, copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score

THESIS    = Path(__file__).resolve().parents[1]
DEVIGN    = THESIS / "devign_full"
CB_DIR    = THESIS / "baselines" / "codebert"
CKPT_DIR  = CB_DIR / "ckpts_mixed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"

MODEL_NAME   = "microsoft/codebert-base"
MAX_LENGTH   = 512
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
PATIENCE     = 3

AUG_FILES = {
    "originals":       DEVIGN / "originals_full_data_with_slices.json",
    "obf_identifier":  DEVIGN / "obf_identifier_full_data_with_slices.json",
    "obf_deadcode":    DEVIGN / "obf_deadcode_full_data_with_slices.json",
    "obf_controlflow": DEVIGN / "obf_controlflow_full_data_with_slices.json",
}
EVAL_FILES = {
    "clean":    DEVIGN / "originals_full_data_with_slices.json",
    "ren":      DEVIGN / "obf_identifier_full_data_with_slices.json",
    "dead":     DEVIGN / "obf_deadcode_full_data_with_slices.json",
    "cf":       DEVIGN / "obf_controlflow_full_data_with_slices.json",
    "ren_dead": DEVIGN / "obf_ren_dead_full_data_with_slices.json",
    "ren_cf":   DEVIGN / "obf_ren_cf_full_data_with_slices.json",
    "dead_cf":  DEVIGN / "obf_dead_cf_full_data_with_slices.json",
    "compound": DEVIGN / "obf_compound_full_data_with_slices.json",
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args   = parser.parse_args()
SEED   = args.seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CodeBERT-Mixed  Seed={SEED}  Device={DEVICE}", flush=True)
set_seed(SEED)

class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r   = self.records[i]
        enc = self.tokenizer(r["code"], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, int(r["label"])

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            out    = model(**batch)
            p      = out.logits.argmax(-1).cpu().tolist()
            preds.extend(p); trues.extend(labels.tolist())
    return f1_score(trues, preds, zero_division=0) * 100

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
split     = json.loads(SPLIT_FILE.read_text())
split     = split.get("splits", split)
train_set = set(split["train"])
test_files = list(split["test"])

# Load: 50% clean + 50% augmented (one randomly sampled transform variant)
# Each epoch we keep a fixed 50/50 split for reproducibility given SEED
print("Loading mixed training data ...", flush=True)
orig_data  = json.loads(Path(AUG_FILES["originals"]).read_text())
orig_train = [r for r in orig_data if r.get("file_name") in train_set]

aug_keys   = ["obf_identifier", "obf_deadcode", "obf_controlflow"]
rng_local  = np.random.default_rng(SEED)
aug_choice = rng_local.choice(aug_keys)
aug_data   = json.loads(Path(AUG_FILES[aug_choice]).read_text())
aug_train  = [r for r in aug_data if r.get("file_name") in train_set]

# Balance: 50% orig, 50% aug (take min to keep equal)
n_each     = min(len(orig_train), len(aug_train))
train_recs = orig_train[:n_each] + aug_train[:n_each]
print(f"  orig={n_each} + aug({aug_choice})={n_each} = {len(train_recs)} total", flush=True)

valid_recs = [r for r in orig_data if r.get("file_name") in set(split["valid"])]


train_ds = CodeDataset(train_recs, tokenizer)
valid_ds = CodeDataset(valid_recs, tokenizer)

labels_list = [r["label"] for r in train_recs]
pos = sum(labels_list); neg = len(labels_list) - pos
weights = [1.0/neg if l == 0 else 1.0/pos for l in labels_list]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

model     = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
n_steps   = len(train_loader) * NUM_EPOCHS
warmup    = int(n_steps * WARMUP_RATIO)
sched     = SequentialLR(optimizer,
    [LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
     ConstantLR(optimizer, factor=1.0, total_iters=n_steps - warmup)],
    milestones=[warmup])

best_val_f1  = -1
best_state   = None
patience_cnt = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss, n_steps_ep = 0, 0
    for batch, labels in train_loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        out    = model(**batch, labels=labels)
        loss   = out.loss
        optimizer.zero_grad(); loss.backward(); optimizer.step(); sched.step()
        total_loss += loss.item(); n_steps_ep += 1
    val_f1 = evaluate(model, valid_loader, DEVICE)
    print(f"  Ep {epoch}/{NUM_EPOCHS} loss={total_loss/n_steps_ep:.4f} val_F1={val_f1:.2f}%",
          flush=True)
    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_state   = copy.deepcopy(model.state_dict())
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}", flush=True); break

model.load_state_dict(best_state)
ckpt_path = CKPT_DIR / f"codebert_mixed_seed{SEED}.pt"
torch.save({"model_state_dict": best_state, "seed": SEED,
            "best_val_f1": best_val_f1}, str(ckpt_path))
print(f"Saved checkpoint → {ckpt_path}", flush=True)

# Evaluate
results = {"seed": SEED, "best_val_f1": round(best_val_f1, 2), "training": "mixed"}
model.eval()
for cond, data_path in EVAL_FILES.items():
    if not Path(data_path).exists():
        continue
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    recs = [idx[f] for f in test_files if f in idx]
    if not recs:
        continue
    ds     = CodeDataset(recs, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    f1 = evaluate(model, loader, DEVICE)
    results[cond] = {"f1": round(f1, 2)}
    print(f"  {cond}: F1={f1:.2f}%", flush=True)

out_path = DEVIGN / f"codebert_mixed_seed{SEED}_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out_path}", flush=True)
