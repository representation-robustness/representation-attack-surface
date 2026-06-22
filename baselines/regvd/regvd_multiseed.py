#!/usr/bin/env python3
"""
ReGVD 5-seed multiseed evaluation on Devign.

Builds graphs once (CodeBERT embeddings are frozen/deterministic),
then trains 5 independent GNN seeds to quantify natural variance.

Outputs:
    ~/thesis/devign_full/regvd_multiseed_results.json
"""

import copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Reuse everything from the existing script
SCRIPT_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_regvd import (
    ReGVDDataset, ReGVDModel,
    focal_loss, evaluate,
    CODEBERT_MODEL, MAX_TOKENS, WINDOW_SIZE,
    HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE,
    LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    DATA_FILES, SPLIT_FILE,
)

THESIS_ROOT = SCRIPT_DIR.parents[1]
DEVIGN_ROOT = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_multiseed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS  = [42, 1337, 7, 100, 999]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_one_seed(seed, train_ds, valid_ds, test_datasets, train_labels):
    set_seed(seed)
    print(f"\n{'='*55}\n  ReGVD Devign  Seed {seed}\n{'='*55}", flush=True)
    ckpt = CKPT_DIR / f"regvd_seed{seed}.pt"

    pos = sum(train_labels); neg = len(train_labels) - pos
    weights = [1.0/neg if l == 0 else 1.0/pos for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    test_loaders = {k: DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
                    for k, ds in test_datasets.items()}

    model     = ReGVDModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0; no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = 0.0; n_steps = 0
        for data in train_loader:
            data = data.to(DEVICE)
            loss = focal_loss(model(data), data.y.view(-1))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item(); n_steps += 1
        _, _, _, vf = evaluate(model, valid_loader, DEVICE)
        print(f"  Ep {epoch:3d}/{NUM_EPOCHS} loss={total_loss/n_steps:.4f} val_F1={vf:.2f}%", flush=True)
        if vf > best_val_f1 + 0.1:
            best_val_f1 = vf; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    results = {'seed': seed, 'best_val_f1': round(best_val_f1, 2)}
    for split_name, loader in test_loaders.items():
        preds, truths = [], []
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                preds.extend(model(data).argmax(-1).cpu().tolist())
                truths.extend(data.y.cpu().tolist())
        f1  = f1_score(truths, preds, zero_division=0) * 100
        pr  = precision_score(truths, preds, zero_division=0) * 100
        rc  = recall_score(truths, preds, zero_division=0) * 100
        acc = accuracy_score(truths, preds) * 100
        results[split_name] = {'f1': round(f1, 2), 'acc': round(acc, 2),
                                'pr': round(pr, 2), 'rc': round(rc, 2)}
        print(f"  {split_name}: F1={f1:.2f}%", flush=True)

    out = DEVIGN_ROOT / f"regvd_seed{seed}_results.json"
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    return results


def aggregate(all_results):
    splits = ['test', 'test_obf_identifier', 'test_obf_deadcode', 'test_obf_controlflow']
    agg = {}
    for s in splits:
        f1s = [r[s]['f1'] for r in all_results if s in r]
        agg[s] = {'f1_mean': round(float(np.mean(f1s)), 2),
                  'f1_std':  round(float(np.std(f1s)), 2), 'all_f1': f1s}
    base = agg['test']['f1_mean']
    for s in splits[1:]:
        agg[s]['delta_f1'] = round(agg[s]['f1_mean'] - base, 2)
    agg.update({'n_seeds': len(all_results), 'seeds': [r['seed'] for r in all_results],
                'model': 'ReGVD', 'dataset': 'Devign'})
    return agg


def main():
    print(f"Device: {DEVICE}", flush=True)

    with open(SPLIT_FILE) as f: split = json.load(f)
    with open(DATA_FILES["originals"]) as f: orig = json.load(f)
    idx = {d["file_name"]: d for d in orig}

    train_recs = [idx[n] for n in split["splits"]["train"] if n in idx]
    valid_recs = [idx[n] for n in split["splits"]["valid"] if n in idx]
    test_recs  = [idx[n] for n in split["splits"]["test"]  if n in idx]

    print(f"Loading CodeBERT embeddings...", flush=True)
    tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
    del codebert

    print("Building graphs (done once for all seeds)...", flush=True)
    train_ds = ReGVDDataset(train_recs, embed_weight, tokenizer)
    valid_ds = ReGVDDataset(valid_recs, embed_weight, tokenizer)

    test_datasets = {'test': ReGVDDataset(test_recs, embed_weight, tokenizer)}
    for obf_key, obf_file in [
        ('test_obf_identifier',  DATA_FILES["obf_identifier"]),
        ('test_obf_deadcode',    DATA_FILES["obf_deadcode"]),
        ('test_obf_controlflow', DATA_FILES["obf_controlflow"]),
    ]:
        with open(obf_file) as f: obf_data = json.load(f)
        obf_idx  = {d["file_name"]: d for d in obf_data}
        obf_recs = [obf_idx[n] for n in split["splits"]["test"] if n in obf_idx]
        test_datasets[obf_key] = ReGVDDataset(obf_recs, embed_weight, tokenizer)

    train_labels = [int(r["label"]) for r in train_recs]
    print(f"Graphs built. train={len(train_ds)} valid={len(valid_ds)}", flush=True)

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, train_ds, valid_ds, test_datasets, train_labels)
        all_results.append(r)

    agg = aggregate(all_results)
    out = DEVIGN_ROOT / "regvd_multiseed_results.json"
    with open(out, 'w') as f: json.dump(agg, f, indent=2)
    print(f"\nResults → {out}", flush=True)
    base = agg['test']['f1_mean']
    print(f"  test:                 F1={agg['test']['f1_mean']:.2f}±{agg['test']['f1_std']:.2f}%")
    for k in ['test_obf_identifier', 'test_obf_deadcode', 'test_obf_controlflow']:
        d = agg[k]
        print(f"  {k}: F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}% Δ={d['delta_f1']:+.2f}pp")


if __name__ == "__main__":
    main()
