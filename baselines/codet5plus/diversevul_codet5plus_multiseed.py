#!/usr/bin/env python3
"""CodeT5+ 220M 5-seed multiseed evaluation on DiverseVul (balanced subset)."""

import copy, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, T5EncoderModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
SPLITS_DIR  = THESIS_ROOT / "diversevul_dataset" / "splits"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_diversevul_codet5plus"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "Salesforce/codet5p-220m"
MAX_LENGTH   = 512
BATCH_SIZE   = 32
LR           = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
PATIENCE     = 2
SEEDS        = [42, 1337, 7, 100, 999]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CONDITIONS = {
    "original":    "test.jsonl",
    "identifier":  "test_obf_identifier.jsonl",
    "deadcode":    "test_obf_deadcode.jsonl",
    "controlflow": "test_obf_controlflow.jsonl",
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": int(r["target"])} for r in rows]


class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records; self.tokenizer = tokenizer

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(rec["code"], max_length=MAX_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(rec["label"], dtype=torch.long)}


def make_loader(records, tokenizer, batch_size, balanced=False):
    ds = CodeDataset(records, tokenizer)
    if balanced:
        labels = [r["label"] for r in records]
        pos = sum(labels); neg = len(labels) - pos
        w = [1.0/neg if l == 0 else 1.0/pos for l in labels]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


class CodeT5PlusClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = T5EncoderModel.from_pretrained(MODEL_NAME)
        hidden          = self.encoder.config.d_model
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, truths = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        preds.extend(logits.argmax(-1).cpu().tolist())
        truths.extend(batch["labels"].tolist())
    return (f1_score(truths, preds, zero_division=0) * 100,
            precision_score(truths, preds, zero_division=0) * 100,
            recall_score(truths, preds, zero_division=0) * 100,
            accuracy_score(truths, preds) * 100)


def train_one_seed(seed, tokenizer, train_recs, valid_recs, test_sets):
    set_seed(seed)
    print(f"\n{'='*55}\n  CodeT5+ DiverseVul  Seed {seed}\n{'='*55}", flush=True)
    ckpt = CKPT_DIR / f"codet5plus_dv_seed{seed}.pt"

    train_loader = make_loader(train_recs, tokenizer, BATCH_SIZE, balanced=True)
    valid_loader = make_loader(valid_recs, tokenizer, BATCH_SIZE * 2)

    model = CodeT5PlusClassifier().to(DEVICE)
    total_steps  = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
         ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)],
        milestones=[warmup_steps])

    best_val_f1 = 0.0; best_state = copy.deepcopy(model.state_dict()); no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = n = 0
        for batch in train_loader:
            iids = batch["input_ids"].to(DEVICE)
            amsk = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            logits = model(iids, amsk)
            loss = nn.functional.cross_entropy(logits, lbls)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); n += 1
        val_f1, _, _, _ = evaluate(model, valid_loader)
        print(f"  Ep {epoch}/{NUM_EPOCHS} loss={total_loss/n:.4f} val_F1={val_f1:.2f}%", flush=True)
        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1; best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    result = {"seed": seed, "best_val_f1": round(best_val_f1, 2)}
    for cond, recs in test_sets.items():
        loader = make_loader(recs, tokenizer, BATCH_SIZE * 2)
        f1, pr, rc, acc = evaluate(model, loader)
        result[cond] = {"f1": round(f1,2), "pr": round(pr,2),
                        "rc": round(rc,2), "acc": round(acc,2)}
        print(f"  {cond:<20} F1={f1:.2f}%", flush=True)
    return result


def main():
    print(f"Device: {DEVICE}  Seeds: {SEEDS}", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_recs = load_jsonl(SPLITS_DIR / "train.jsonl")
    valid_recs = load_jsonl(SPLITS_DIR / "valid.jsonl")
    test_sets  = {k: load_jsonl(SPLITS_DIR / v) for k, v in TEST_CONDITIONS.items()
                  if (SPLITS_DIR / v).exists()}
    n_pos = sum(r["label"] for r in train_recs)
    print(f"train={len(train_recs)} ({n_pos} pos)  valid={len(valid_recs)}", flush=True)

    all_results = [train_one_seed(s, tokenizer, train_recs, valid_recs, test_sets)
                   for s in SEEDS]

    conds = list(TEST_CONDITIONS.keys())
    agg = {"n_seeds": len(SEEDS), "seeds": SEEDS, "model": "CodeT5+", "dataset": "DiverseVul"}
    for cond in conds:
        f1s = [r[cond]["f1"] for r in all_results if cond in r]
        agg[cond] = {"f1_mean": round(float(np.mean(f1s)), 2),
                     "f1_std":  round(float(np.std(f1s)),  2), "all_f1": f1s}
    base = agg["original"]["f1_mean"]
    for cond in conds[1:]:
        agg[cond]["delta_f1"] = round(agg[cond]["f1_mean"] - base, 2)

    out = RESULTS_DIR / "diversevul_codet5plus_multiseed_results.json"
    with open(out, "w") as f: json.dump(agg, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
    for cond in conds:
        d = agg[cond]
        delta = f"  Δ={d.get('delta_f1',0):+.2f}" if cond != "original" else ""
        print(f"  {cond:<20} F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}{delta}")


if __name__ == "__main__":
    main()
