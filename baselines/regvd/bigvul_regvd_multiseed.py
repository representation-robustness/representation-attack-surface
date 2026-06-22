#!/usr/bin/env python3
"""ReGVD 5-seed multiseed evaluation on Big-Vul."""

import copy, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
SPLITS_DIR  = THESIS_ROOT / "bigvul" / "splits"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_multiseed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from train_regvd import (
    ReGVDDataset, ReGVDModel, focal_loss, evaluate,
    CODEBERT_MODEL, MAX_TOKENS, WINDOW_SIZE,
    HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE,
    LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
)

SEEDS = [42, 1337, 7, 100, 999]
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


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": int(r["target"])} for r in rows]


def train_one_seed(seed, train_ds, valid_ds, test_datasets, train_labels):
    set_seed(seed)
    print(f"\n{'='*55}\n  ReGVD BigVul  Seed {seed}\n{'='*55}", flush=True)

    pos = sum(train_labels); neg = len(train_labels) - pos
    weights = [1.0/neg if l == 0 else 1.0/pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    test_loaders = {k: DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
                    for k, ds in test_datasets.items()}

    model     = ReGVDModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0; no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = 0.0; n = 0
        for data in train_loader:
            data = data.to(DEVICE)
            loss = focal_loss(model(data), data.y.view(-1))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item(); n += 1
        _, _, _, vf = evaluate(model, valid_loader, DEVICE)
        print(f"  Ep {epoch:3d}/{NUM_EPOCHS} loss={total_loss/n:.4f} val_F1={vf:.2f}%", flush=True)
        if vf > best_val_f1 + 0.1:
            best_val_f1 = vf; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    result = {"seed": seed, "best_val_f1": round(best_val_f1, 2)}
    model.eval()
    for cond, loader in test_loaders.items():
        preds, truths = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                preds.extend(model(data).argmax(-1).cpu().tolist())
                truths.extend(data.y.cpu().tolist())
        f1  = f1_score(truths, preds, zero_division=0) * 100
        pr  = precision_score(truths, preds, zero_division=0) * 100
        rc  = recall_score(truths, preds, zero_division=0) * 100
        acc = accuracy_score(truths, preds) * 100
        result[cond] = {"f1": round(f1,2), "pr": round(pr,2),
                        "rc": round(rc,2), "acc": round(acc,2)}
        print(f"  {cond:<20} F1={f1:.2f}%", flush=True)
    return result


def main():
    print(f"Device: {DEVICE}  Seeds: {SEEDS}", flush=True)

    print("Loading CodeBERT embeddings...", flush=True)
    tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
    del codebert

    print("Loading splits...", flush=True)
    train_recs = load_jsonl(SPLITS_DIR / "train.jsonl")
    valid_recs = load_jsonl(SPLITS_DIR / "valid.jsonl")
    n_pos = sum(r["label"] for r in train_recs)
    print(f"  train={len(train_recs)} ({n_pos} pos)  valid={len(valid_recs)}", flush=True)

    print("Building graphs (once for all seeds)...", flush=True)
    train_ds = ReGVDDataset(train_recs, embed_weight, tokenizer)
    valid_ds = ReGVDDataset(valid_recs, embed_weight, tokenizer)

    test_datasets = {}
    for cond, fname in TEST_CONDITIONS.items():
        path = SPLITS_DIR / fname
        if path.exists():
            test_datasets[cond] = ReGVDDataset(load_jsonl(path), embed_weight, tokenizer)
    print(f"Graphs built.", flush=True)

    train_labels = [int(r["label"]) for r in train_recs]
    all_results  = [train_one_seed(s, train_ds, valid_ds, test_datasets, train_labels)
                    for s in SEEDS]

    conds = list(test_datasets.keys())
    agg = {"n_seeds": len(SEEDS), "seeds": SEEDS, "model": "ReGVD", "dataset": "Big-Vul"}
    for cond in conds:
        f1s = [r[cond]["f1"] for r in all_results if cond in r]
        agg[cond] = {"f1_mean": round(float(np.mean(f1s)), 2),
                     "f1_std":  round(float(np.std(f1s)),  2), "all_f1": f1s}
    base = agg["original"]["f1_mean"]
    for cond in conds[1:]:
        agg[cond]["delta_f1"] = round(agg[cond]["f1_mean"] - base, 2)

    out = RESULTS_DIR / "bigvul_regvd_multiseed_results.json"
    with open(out, "w") as f: json.dump(agg, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
    print(f"  original  F1={agg['original']['f1_mean']:.2f}±{agg['original']['f1_std']:.2f}")
    for cond in conds[1:]:
        d = agg[cond]
        print(f"  {cond:<20} F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}  Δ={d['delta_f1']:+.2f}")


if __name__ == "__main__":
    main()
