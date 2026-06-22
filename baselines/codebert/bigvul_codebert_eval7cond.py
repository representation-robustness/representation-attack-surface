#!/usr/bin/env python3
"""Eval-only: load existing BigVul CodeBERT checkpoints and eval 7 conditions."""

import json, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
SPLITS_DIR  = THESIS_ROOT / "bigvul" / "splits"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_multiseed"

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 32
SEEDS      = [42, 1337, 7, 100, 999]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CONDITIONS = {
    "original":    "test.jsonl",
    "identifier":  "test_obf_identifier.jsonl",
    "deadcode":    "test_obf_deadcode.jsonl",
    "controlflow": "test_obf_controlflow.jsonl",
    "ren_dead":    "test_obf_ren_dead.jsonl",
    "ren_cf":      "test_obf_ren_cf.jsonl",
    "dead_cf":     "test_obf_dead_cf.jsonl",
    "compound":    "test_obf_compound.jsonl",
}


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": int(r["target"])} for r in rows]


class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records; self.tokenizer = tokenizer

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(r["code"], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(r["label"], dtype=torch.long)}


def make_loader(records, tokenizer, batch_size):
    return DataLoader(CodeDataset(records, tokenizer), batch_size=batch_size,
                      shuffle=False, num_workers=4, pin_memory=True)


def evaluate(model, loader):
    model.eval(); preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            iids = batch["input_ids"].to(DEVICE)
            amsk = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=iids, attention_mask=amsk).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            truths.extend(batch["labels"].tolist())
    return (f1_score(truths, preds, zero_division=0) * 100,
            precision_score(truths, preds, zero_division=0) * 100,
            recall_score(truths, preds, zero_division=0) * 100,
            accuracy_score(truths, preds) * 100)


def eval_one_seed(seed, tokenizer, test_sets):
    ckpt = CKPT_DIR / f"codebert_bigvul_seed{seed}.pt"
    print(f"\n  Seed {seed}  ckpt={ckpt.name}", flush=True)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    result = {"seed": seed}
    for cond, recs in test_sets.items():
        loader = make_loader(recs, tokenizer, BATCH_SIZE)
        f1, pr, rc, acc = evaluate(model, loader)
        result[cond] = {"f1": round(f1, 2), "pr": round(pr, 2),
                        "rc": round(rc, 2), "acc": round(acc, 2)}
        print(f"    {cond:<20} F1={f1:.2f}%", flush=True)
    return result


def main():
    print(f"Device: {DEVICE}  Seeds: {SEEDS}", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    test_sets = {k: load_jsonl(SPLITS_DIR / v) for k, v in TEST_CONDITIONS.items()
                 if (SPLITS_DIR / v).exists()}
    print(f"Test conditions: {list(test_sets.keys())}", flush=True)

    all_results = [eval_one_seed(s, tokenizer, test_sets) for s in SEEDS]

    conds = [c for c in TEST_CONDITIONS if c in test_sets]
    agg = {"n_seeds": len(SEEDS), "seeds": SEEDS, "model": "CodeBERT", "dataset": "Big-Vul"}
    for cond in conds:
        f1s  = [r[cond]["f1"]  for r in all_results if cond in r]
        accs = [r[cond]["acc"] for r in all_results if cond in r]
        prs  = [r[cond]["pr"]  for r in all_results if cond in r]
        rcs  = [r[cond]["rc"]  for r in all_results if cond in r]
        agg[cond] = {"f1_mean": round(float(np.mean(f1s)), 2),
                     "f1_std":  round(float(np.std(f1s)),  2),
                     "acc_mean": round(float(np.mean(accs)), 2),
                     "pr_mean":  round(float(np.mean(prs)),  2),
                     "rc_mean":  round(float(np.mean(rcs)),  2),
                     "all_f1": f1s}
    base = agg["original"]["f1_mean"]
    for cond in conds[1:]:
        agg[cond]["delta_f1"] = round(agg[cond]["f1_mean"] - base, 2)

    out = RESULTS_DIR / "bigvul_codebert_multiseed_results.json"
    with open(out, "w") as f: json.dump(agg, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
    for cond in conds:
        d = agg[cond]
        delta = f"  Δ={d.get('delta_f1',0):+.2f}" if cond != "original" else ""
        print(f"  {cond:<20} F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}{delta}")


if __name__ == "__main__":
    main()
