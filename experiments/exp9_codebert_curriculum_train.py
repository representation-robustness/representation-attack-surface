#!/usr/bin/env python3
"""
Exp 9: CodeBERT curriculum training.
    Phase 1 (epochs 1-2): clean data only
    Phase 2 (epochs 3-4): clean + renaming (2x)
    Phase 3 (epochs 5-7): clean + renaming + deadcode (3x)
    Phase 4 (epochs 8-10): clean + all 3 transforms (4x, same as full aug)

Run 5 instances in parallel, one per GPU. THIS IS FOR THE PEER TO RUN TOMORROW.

Usage:
    CUDA_VISIBLE_DEVICES=X python exp9_codebert_curriculum_train.py --seed SEED

where SEED is one of: 42, 1337, 7, 100, 999

Output:
    baselines/codebert/ckpts_curriculum/codebert_curriculum_seedSEED.pt
    devign_full/codebert_curriculum_seedSEED_results.json

After all 5 seeds complete, run exp9_aggregate.py (also in this folder) to combine.
"""

import argparse, copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score

THESIS    = Path(__file__).resolve().parents[1]
DEVIGN    = THESIS / "devign_full"
CB_DIR    = THESIS / "baselines" / "codebert"
CKPT_DIR  = CB_DIR / "ckpts_curriculum"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"

MODEL_NAME   = "microsoft/codebert-base"
MAX_LENGTH   = 512
BATCH_SIZE   = 16
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
PATIENCE     = 3

# Curriculum phases: (n_epochs, [data_keys])
CURRICULUM = [
    (2, ["originals"]),
    (2, ["originals", "obf_identifier"]),
    (3, ["originals", "obf_identifier", "obf_deadcode"]),
    (3, ["originals", "obf_identifier", "obf_deadcode", "obf_controlflow"]),
]

DATA_PATHS = {
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
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            trues.extend(labels.tolist())
    return f1_score(trues, preds, zero_division=0) * 100

def make_loader(recs, tokenizer, batch_size, is_train=True):
    ds = CodeDataset(recs, tokenizer)
    if is_train:
        labels = [r["label"] for r in recs]
        pos = max(sum(labels), 1); neg = max(len(labels) - pos, 1)
        w = [1.0/neg if l == 0 else 1.0/pos for l in labels]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size * 2, shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args   = parser.parse_args()
SEED   = args.seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CodeBERT-Curriculum  Seed={SEED}  Device={DEVICE}", flush=True)
set_seed(SEED)

tokenizer  = RobertaTokenizer.from_pretrained(MODEL_NAME)
split      = json.loads(SPLIT_FILE.read_text())
split      = split.get("splits", split)
train_set  = set(split["train"])
test_files = list(split["test"])

# Preload all data files
all_data = {}
for k, p in DATA_PATHS.items():
    d = json.loads(Path(p).read_text())
    all_data[k] = {r["file_name"]: r for r in d}

orig_data  = json.loads(Path(DATA_PATHS["originals"]).read_text())
valid_recs = [r for r in orig_data if r.get("file_name") in set(split["valid"])]

valid_loader = make_loader(valid_recs, tokenizer, BATCH_SIZE, is_train=False)

total_epochs = sum(n for n, _ in CURRICULUM)
model     = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
n_steps_total = total_epochs * (len(train_set) // BATCH_SIZE + 1)
warmup    = int(n_steps_total * WARMUP_RATIO)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched     = SequentialLR(optimizer,
    [LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
     ConstantLR(optimizer, factor=1.0, total_iters=max(n_steps_total - warmup, 1))],
    milestones=[warmup])

best_val_f1  = -1
best_state   = None
patience_cnt = 0
global_epoch = 0

for phase_idx, (n_epochs, data_keys) in enumerate(CURRICULUM):
    # Build training set for this phase
    phase_recs = []
    for k in data_keys:
        recs = [all_data[k][f] for f in train_set if f in all_data[k]]
        phase_recs.extend(recs)
    print(f"\nPhase {phase_idx+1}: keys={data_keys}, n={len(phase_recs)}", flush=True)
    train_loader = make_loader(phase_recs, tokenizer, BATCH_SIZE, is_train=True)

    for ep in range(n_epochs):
        global_epoch += 1
        model.train()
        total_loss, nsteps = 0, 0
        for batch, labels in train_loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)
            out    = model(**batch, labels=labels)
            optimizer.zero_grad(); out.loss.backward(); optimizer.step(); sched.step()
            total_loss += out.loss.item(); nsteps += 1
        val_f1 = evaluate(model, valid_loader, DEVICE)
        print(f"  Ep {global_epoch:2d} (phase {phase_idx+1}) loss={total_loss/nsteps:.4f} "
              f"val_F1={val_f1:.2f}%", flush=True)
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at ep {global_epoch}", flush=True)
                break
    else:
        continue
    break

model.load_state_dict(best_state)
ckpt_path = CKPT_DIR / f"codebert_curriculum_seed{SEED}.pt"
torch.save({"model_state_dict": best_state, "seed": SEED,
            "best_val_f1": best_val_f1}, str(ckpt_path))
print(f"Saved checkpoint → {ckpt_path}", flush=True)

# Evaluate on all 7 conditions
results = {"seed": SEED, "best_val_f1": round(best_val_f1, 2), "training": "curriculum"}
model.eval()
for cond, data_path in EVAL_FILES.items():
    if not Path(data_path).exists(): continue
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    recs = [idx[f] for f in test_files if f in idx]
    if not recs: continue
    loader = make_loader(recs, tokenizer, BATCH_SIZE, is_train=False)
    f1 = evaluate(model, loader, DEVICE)
    results[cond] = {"f1": round(f1, 2)}
    print(f"  {cond}: F1={f1:.2f}%", flush=True)

out_path = DEVIGN / f"codebert_curriculum_seed{SEED}_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out_path}", flush=True)
