#!/usr/bin/env python3
"""
ECG RGCN 5-seed multiseed on Devign (5-edge GGNN format).

Graphs are pre-built JSON — no Joern needed.
Outputs:
    ~/thesis/devign_full/ecgrgcn_multiseed_results.json
"""

import copy, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

SCRIPT_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_rgcn import (
    load_graphs, balanced_loader, RelationalGNN, focal_loss,
    predict, best_f1_threshold, eval_metrics,
    HIDDEN, NUM_LAYERS, NUM_RELATIONS, BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE,
)

THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_ROOT  = THESIS_ROOT / "devign_full"
DEVIGN_INPUT = DEVIGN_ROOT / "devign_input"

SEEDS  = [42, 1337, 7, 100, 999]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SPLITS = {
    "test":                 DEVIGN_INPUT / "originals_train/test_GGNNinput.json",
    "test_obf_identifier":  DEVIGN_INPUT / "obf_identifier_test/test_GGNNinput.json",
    "test_obf_deadcode":    DEVIGN_INPUT / "obf_deadcode_test/test_GGNNinput.json",
    "test_obf_controlflow": DEVIGN_INPUT / "obf_controlflow_test/test_GGNNinput.json",
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_one_seed(seed, train_graphs, valid_loader, test_loaders, in_dim):
    set_seed(seed)
    print(f"\n{'='*55}\n  ECG RGCN Devign  Seed {seed}\n{'='*55}", flush=True)

    model     = RelationalGNN(in_channels=in_dim, hidden=HIDDEN,
                              num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0; best_thr = 0.5; no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); total_loss = 0.0; n_batches = 0
        loader = balanced_loader(train_graphs, BATCH_SIZE)
        for b in loader:
            b = b.to(DEVICE)
            loss = focal_loss(model(b.x, b.edge_index, b.edge_type, b.batch), b.y.squeeze(-1))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); total_loss += loss.item(); n_batches += 1
        scheduler.step()

        val_logits, val_labels = predict(model, valid_loader, DEVICE)
        thr, val_f1 = best_f1_threshold(val_logits, val_labels)
        val_f1 *= 100
        print(f"  Ep {epoch:3d}/{MAX_EPOCHS} loss={total_loss/max(n_batches,1):.4f} "
              f"val_F1={val_f1:.2f}% thr={thr:.2f}", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1; best_thr = thr
            best_state  = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop ep {epoch}", flush=True); break

    model.load_state_dict(best_state)
    ckpt_dir = os.path.expanduser('~/ecgrgcn_devign_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, f'ecgrgcn_dv_seed{seed}.pt'))
    results = {'seed': seed, 'best_val_f1': round(best_val_f1, 2), 'best_thr': best_thr}
    for split_name, loader in test_loaders.items():
        logits, labels = predict(model, loader, DEVICE)
        acc, pr, rc, f1 = eval_metrics(logits, labels, best_thr)
        results[split_name] = {'f1': round(f1, 2), 'acc': round(acc, 2),
                                'pr': round(pr, 2),  'rc': round(rc, 2)}
        print(f"  {split_name}: F1={f1:.2f}%", flush=True)

    out = DEVIGN_ROOT / f"ecgrgcn_seed{seed}_results.json"
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    return results


def aggregate(all_results):
    splits = list(TEST_SPLITS.keys())
    agg = {}
    for s in splits:
        f1s = [r[s]['f1'] for r in all_results if s in r]
        agg[s] = {'f1_mean': round(float(np.mean(f1s)), 2),
                  'f1_std':  round(float(np.std(f1s)), 2), 'all_f1': f1s}
    base = agg['test']['f1_mean']
    for s in splits[1:]:
        agg[s]['delta_f1'] = round(agg[s]['f1_mean'] - base, 2)
    agg.update({'n_seeds': len(all_results), 'seeds': [r['seed'] for r in all_results],
                'model': 'ECG RGCN', 'dataset': 'Devign'})
    return agg


def main():
    print(f"Device: {DEVICE}", flush=True)

    print("Loading graphs...", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train/train_GGNNinput.json")
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train/valid_GGNNinput.json")
    in_dim       = train_graphs[0].x.shape[1]
    print(f"  train={len(train_graphs)} valid={len(valid_graphs)} in_dim={in_dim}", flush=True)

    valid_loader = DataLoader(valid_graphs, batch_size=128, shuffle=False)
    test_loaders = {k: DataLoader(load_graphs(p), batch_size=128, shuffle=False)
                    for k, p in TEST_SPLITS.items()}

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, train_graphs, valid_loader, test_loaders, in_dim)
        all_results.append(r)

    agg = aggregate(all_results)
    out = DEVIGN_ROOT / "ecgrgcn_multiseed_results.json"
    with open(out, 'w') as f: json.dump(agg, f, indent=2)
    print(f"\nResults → {out}", flush=True)
    base = agg['test']['f1_mean']
    print(f"  test:               F1={agg['test']['f1_mean']:.2f}±{agg['test']['f1_std']:.2f}%")
    for k in list(TEST_SPLITS.keys())[1:]:
        d = agg[k]
        print(f"  {k}: F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}% Δ={d['delta_f1']:+.2f}pp")


if __name__ == "__main__":
    main()
