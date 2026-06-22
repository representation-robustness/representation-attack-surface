#!/usr/bin/env python3
"""
CodeT5+ 220M single-seed evaluation on Big-Vul (cross-dataset generalisation).

Reads from ~/thesis/bigvul/splits/{train,valid,test,test_obf_*}.jsonl
Each line: {"func": "<code>", "target": 0|1, ...}

Output: ~/thesis/devign_full/bigvul_codet5plus_results.json
"""

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

THESIS_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR  = THESIS_ROOT / "bigvul" / "splits"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_PATH   = Path(__file__).resolve().parent / "ckpts_codet5plus" / "codet5plus_bigvul.pt"

MODEL_NAME   = "Salesforce/codet5p-220m"
MAX_LENGTH   = 512
BATCH_SIZE   = 32
LR           = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS   = 5
WARMUP_RATIO = 0.1
PATIENCE     = 2
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": r["target"]} for r in rows]


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
                "labels":         torch.tensor(int(rec["label"]), dtype=torch.long)}


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
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, truths = [], []
    for batch in loader:
        iids = batch["input_ids"].to(DEVICE)
        amsk = batch["attention_mask"].to(DEVICE)
        preds.extend(model(iids, amsk).argmax(-1).cpu().tolist())
        truths.extend(batch["labels"].tolist())
    return (f1_score(truths, preds, zero_division=0) * 100,
            accuracy_score(truths, preds) * 100,
            precision_score(truths, preds, zero_division=0) * 100,
            recall_score(truths, preds, zero_division=0) * 100)


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "", flush=True)

    print("Loading splits...", flush=True)
    train_recs = load_jsonl(SPLITS_DIR / "train.jsonl")
    valid_recs = load_jsonl(SPLITS_DIR / "valid.jsonl")
    n_pos = sum(r["label"] for r in train_recs)
    print(f"  train={len(train_recs)} ({n_pos} pos, {len(train_recs)-n_pos} neg) "
          f"valid={len(valid_recs)}", flush=True)

    test_conditions = {
        "original":    "test.jsonl",
        "identifier":  "test_obf_identifier.jsonl",
        "deadcode":    "test_obf_deadcode.jsonl",
        "controlflow": "test_obf_controlflow.jsonl",
    }

    print(f"Loading tokenizer {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    print(f"\nTraining (seed={SEED}, {NUM_EPOCHS} epochs, patience={PATIENCE})...", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = 0; n_steps = 0
        for batch in train_loader:
            iids = batch["input_ids"].to(DEVICE)
            amsk = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            logits = model(iids, amsk)
            loss = nn.functional.cross_entropy(logits, lbls)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); n_steps += 1

        val_f1, _, _, _ = evaluate(model, valid_loader)
        print(f"  Ep {epoch}/{NUM_EPOCHS} loss={total_loss/n_steps:.4f} "
              f"val_F1={val_f1:.2f}%", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1; best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_PATH); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    print(f"\nBest val F1: {best_val_f1:.2f}%", flush=True)

    results = {"dataset": "Big-Vul", "model": "CodeT5+", "seed": SEED,
               "best_val_f1": round(best_val_f1, 2), "results": {}}

    for cond, fname in test_conditions.items():
        fpath = SPLITS_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {cond} (file not found)", flush=True); continue
        test_recs   = load_jsonl(fpath)
        test_loader = make_loader(test_recs, tokenizer, BATCH_SIZE * 2)
        f1, acc, pr, rc = evaluate(model, test_loader)
        results["results"][cond] = {"f1": round(f1, 2), "acc": round(acc, 2),
                                    "pr": round(pr, 2), "rc": round(rc, 2)}
        print(f"  {cond:<20} F1={f1:.2f}% Pr={pr:.2f}% Rc={rc:.2f}%", flush=True)

    base = results["results"].get("original", {}).get("f1", 0)
    for cond in results["results"]:
        if cond != "original":
            results["results"][cond]["delta_f1"] = round(
                results["results"][cond]["f1"] - base, 2)

    out = RESULTS_DIR / "bigvul_codet5plus_results.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
    print(f"  original:    F1={base:.2f}%")
    for cond in ["identifier", "deadcode", "controlflow"]:
        if cond in results["results"]:
            d = results["results"][cond]
            print(f"  {cond:<20} F1={d['f1']:.2f}% Δ={d['delta_f1']:+.2f}pp")


if __name__ == "__main__":
    main()
