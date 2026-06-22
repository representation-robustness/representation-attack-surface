#!/usr/bin/env python3
"""
Exp 6: ReGVD augmented training (clean + all 3 transforms = 4x data).
Run 5 instances in parallel, one per GPU, each with a different seed.

Usage:
    CUDA_VISIBLE_DEVICES=X python exp6_regvd_aug_train.py --seed SEED

Output:
    baselines/regvd/ckpts_aug/regvd_aug_seedSEED.pt
    devign_full/regvd_aug_seedSEED_results.json

After all 5 seeds done, aggregate with:
    python exp6_regvd_aug_aggregate.py
"""

import argparse, copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

THESIS     = Path(__file__).resolve().parents[1]
DEVIGN     = THESIS / "devign_full"
REGVD_DIR  = THESIS / "baselines" / "regvd"
CKPT_DIR   = REGVD_DIR / "ckpts_aug"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"

sys.path.insert(0, str(REGVD_DIR))
from train_regvd import (
    ReGVDDataset, ReGVDModel, focal_loss, evaluate,
    CODEBERT_MODEL, HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE,
    LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
)

AUG_DATA_FILES = {
    "originals":       DEVIGN / "originals_full_data_with_slices.json",
    "obf_identifier":  DEVIGN / "obf_identifier_full_data_with_slices.json",
    "obf_deadcode":    DEVIGN / "obf_deadcode_full_data_with_slices.json",
    "obf_controlflow": DEVIGN / "obf_controlflow_full_data_with_slices.json",
}
EVAL_DATA_FILES = {
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
args = parser.parse_args()
SEED = args.seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Seed={SEED}  Device={DEVICE}", flush=True)
set_seed(SEED)

# Load splits
split       = json.loads(SPLIT_FILE.read_text())
split       = split.get("splits", split)
train_files = set(split["train"])
valid_files = set(split["valid"])
test_files  = list(split["test"])

# Load CodeBERT embedding weights
from transformers import RobertaTokenizer, RobertaModel
print(f"Loading CodeBERT embedding weights from {CODEBERT_MODEL}...", flush=True)
tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
codebert_tmp = RobertaModel.from_pretrained(CODEBERT_MODEL)
embed_weight = codebert_tmp.embeddings.word_embeddings.weight.detach().cpu()
del codebert_tmp
print(f"  Vocab: {embed_weight.shape[0]:,}  dim: {embed_weight.shape[1]}", flush=True)

# Load and combine augmented training data
print("Loading augmented training data ...", flush=True)
train_recs = []
for name, path in AUG_DATA_FILES.items():
    data = json.loads(Path(path).read_text())
    recs = [r for r in data if r.get("file_name") in train_files]
    train_recs.extend(recs)
    print(f"  {name}: {len(recs)} train records", flush=True)
print(f"Total augmented train: {len(train_recs)}", flush=True)

# Validation (clean only)
orig_data  = json.loads(Path(AUG_DATA_FILES["originals"]).read_text())
orig_idx   = {r["file_name"]: r for r in orig_data}
valid_recs = [orig_idx[f] for f in valid_files if f in orig_idx]
print(f"Valid: {len(valid_recs)}", flush=True)

# Build datasets
print("Building datasets ...", flush=True)
train_ds = ReGVDDataset(train_recs, embed_weight, tokenizer)
valid_ds = ReGVDDataset(valid_recs, embed_weight, tokenizer)

# Weighted sampler for class balance
train_labels = [int(r["label"]) for r in train_recs]
pos = sum(train_labels); neg = len(train_labels) - pos
weights  = [1.0/max(neg,1) if l == 0 else 1.0/max(pos,1) for l in train_labels]
sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

# Model
model     = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM,
                       num_layers=NUM_GNN_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_f1  = -1
best_state   = None
patience_cnt = 0

print(f"\nTraining ReGVD-Aug (seed={SEED}) ...", flush=True)
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch  = batch.to(DEVICE)
        logits = model(batch)
        loss   = focal_loss(logits, batch.y.squeeze().long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    val_acc, val_pr, val_rc, val_f1 = evaluate(model, valid_loader, DEVICE)
    print(f"  Ep {epoch:3d}/{NUM_EPOCHS} loss={total_loss/len(train_loader):.4f} "
          f"val_F1={val_f1:.2f}%", flush=True)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = copy.deepcopy(model.state_dict())
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}", flush=True)
            break

model.load_state_dict(best_state)
ckpt_path = CKPT_DIR / f"regvd_aug_seed{SEED}.pt"
torch.save(best_state, str(ckpt_path))
print(f"Saved checkpoint → {ckpt_path}", flush=True)

# Evaluate on all 7 conditions
print("\nEvaluating on test conditions ...", flush=True)
model.eval()
results = {"seed": SEED, "best_val_f1": round(best_val_f1, 2)}
for cond, data_path in EVAL_DATA_FILES.items():
    if not Path(data_path).exists():
        continue
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    recs = [idx[f] for f in test_files if f in idx]
    if not recs:
        continue
    ds     = ReGVDDataset(recs, embed_weight, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    acc, pr, rc, f1 = evaluate(model, loader, DEVICE)
    results[cond] = {"f1": round(f1, 2), "acc": round(acc, 2),
                     "pr": round(pr, 2),  "rc": round(rc, 2)}
    print(f"  {cond}: F1={f1:.2f}%", flush=True)

out_path = DEVIGN / f"regvd_aug_seed{SEED}_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved results → {out_path}", flush=True)
