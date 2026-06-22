#!/usr/bin/env python3
"""
Train ReGVD seed=42 on BigVul or DiverseVul, save best checkpoint + ROC probabilities.
Usage:
  CUDA_VISIBLE_DEVICES=2 python train_regvd_roc.py bigvul
  CUDA_VISIBLE_DEVICES=5 python train_regvd_roc.py diversevul
"""
import copy, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
from train_regvd import (
    ReGVDDataset, ReGVDModel, focal_loss, evaluate,
    CODEBERT_MODEL, MAX_TOKENS, WINDOW_SIZE,
    HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE,
    LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
)

assert len(sys.argv) == 2 and sys.argv[1] in ("bigvul", "diversevul"), \
    "Usage: python train_regvd_roc.py [bigvul|diversevul]"
DATASET = sys.argv[1]

if DATASET == "bigvul":
    SPLITS_DIR = THESIS_ROOT / "bigvul" / "splits"
else:
    SPLITS_DIR = THESIS_ROOT / "diversevul_dataset" / "splits"

CKPT_PATH = SCRIPT_DIR / f"models/regvd_{DATASET}/best.pt"
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
PROBS_PATH = THESIS_ROOT / "devign_full" / f"{DATASET}_regvd_roc_probs.json"

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dataset: {DATASET}  Device: {DEVICE}  SPLITS: {SPLITS_DIR}", flush=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": int(r["target"])} for r in rows]


def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            logits = model(data)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(data.y.cpu().numpy().tolist())
    return probs, labels


set_seed(SEED)

print("Loading CodeBERT embeddings...", flush=True)
tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
del codebert

print("Loading train/valid splits...", flush=True)
train_recs = load_jsonl(SPLITS_DIR / "train.jsonl")
valid_recs = load_jsonl(SPLITS_DIR / "valid.jsonl")
n_pos      = sum(r["label"] for r in train_recs)
print(f"  train={len(train_recs)} ({n_pos} pos)  valid={len(valid_recs)}", flush=True)

train_ds = ReGVDDataset(train_recs, embed_weight, tokenizer)
valid_ds = ReGVDDataset(valid_recs, embed_weight, tokenizer)

print("Loading test splits...", flush=True)
clean_recs = load_jsonl(SPLITS_DIR / "test.jsonl")
ren_recs   = load_jsonl(SPLITS_DIR / "test_obf_identifier.jsonl")
clean_ds = ReGVDDataset(clean_recs, embed_weight, tokenizer)
ren_ds   = ReGVDDataset(ren_recs,   embed_weight, tokenizer)
print(f"  clean test={len(clean_recs)}  renamed test={len(ren_recs)}", flush=True)

# weighted sampler for class imbalance
train_labels = [r["label"] for r in train_recs]
pos = sum(train_labels); neg = len(train_labels) - pos
weights = [1.0/neg if l == 0 else 1.0/pos for l in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,   sampler=sampler, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE*2, shuffle=False,   num_workers=0)
clean_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE*2, shuffle=False,   num_workers=0)
ren_loader   = DataLoader(ren_ds,   batch_size=BATCH_SIZE*2, shuffle=False,   num_workers=0)

import torch.nn as nn
model     = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_state   = copy.deepcopy(model.state_dict())
best_val_f1  = 0.0
no_improve   = 0

print(f"\nTraining ReGVD on {DATASET} (seed=42) ...", flush=True)
for epoch in range(1, NUM_EPOCHS + 1):
    model.train(); total_loss = 0.0; n = 0
    for data in train_loader:
        data = data.to(DEVICE)
        loss = focal_loss(model(data), data.y.view(-1))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); total_loss += loss.item(); n += 1
    _, _, _, vf = evaluate(model, valid_loader, DEVICE)
    print(f"  Ep {epoch:3d}/{NUM_EPOCHS}  loss={total_loss/n:.4f}  val_F1={vf:.2f}%", flush=True)
    if vf > best_val_f1 + 0.1:
        best_val_f1 = vf
        best_state  = copy.deepcopy(model.state_dict())
        no_improve  = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}", flush=True)
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), CKPT_PATH)
print(f"\nSaved checkpoint → {CKPT_PATH}", flush=True)

clean_probs, clean_labels = get_probs(model, clean_loader)
ren_probs,   ren_labels   = get_probs(model, ren_loader)

clean_f1 = f1_score(clean_labels, [1 if p >= 0.5 else 0 for p in clean_probs], zero_division=0) * 100
ren_f1   = f1_score(ren_labels,   [1 if p >= 0.5 else 0 for p in ren_probs],   zero_division=0) * 100
print(f"  clean test F1={clean_f1:.2f}%  n={len(clean_probs)}", flush=True)
print(f"  renamed test F1={ren_f1:.2f}%  n={len(ren_probs)}", flush=True)

out = {
    "dataset": DATASET,
    "clean":   {"probs": clean_probs, "labels": clean_labels},
    "renamed": {"probs": ren_probs,   "labels": ren_labels},
}
with open(PROBS_PATH, "w") as f:
    json.dump(out, f)
print(f"Saved probabilities → {PROBS_PATH}", flush=True)
