"""
train.py — Train and evaluate VulGNN on Devign.

Usage:
    CUDA_VISIBLE_DEVICES=2 python train.py

Outputs:
    ~/vulgnn_devign/vulgnn_devign.pt     — best checkpoint (by val F1)
    ~/thesis/devign_full/vulgnn_results.json
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, NUM_NODE_TYPES, NUM_EDGE_TYPES
from model import VulGNN

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS    = 25
LR        = 1e-3
BATCH     = 128           # paper uses 400, smaller for memory safety
HIDDEN    = 128
NUM_LAYERS= 6
DROPOUT   = 0.08
EMBED_DIM = 16
EDGE_DIM  = 4

CKPT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vulgnn_devign.pt')
RESULTS_OUT = os.path.expanduser('~/thesis/devign_full/vulgnn_results.json')

SPLITS = {
    'test':                 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
}


def compute_class_weights(loader):
    """Compute pos/neg class weights for weighted cross-entropy."""
    labels = []
    for batch in loader:
        labels.extend(batch.y.tolist())
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # weight[0]=neg weight, weight[1]=pos weight
    total = n_pos + n_neg
    w = torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float)
    return w.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            pred = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(pred)
            truths.extend(batch.y.cpu().tolist())
    f1  = f1_score(truths, preds, zero_division=0)
    acc = accuracy_score(truths, preds)
    return f1, acc


def main():
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading train split...")
    t0 = time.time()
    train_graphs = load_split('train')
    print(f"  {len(train_graphs)} graphs in {time.time()-t0:.1f}s")

    print("Loading valid split...")
    valid_graphs = load_split('valid')
    print(f"  {len(valid_graphs)} graphs")

    print("Loading test splits...")
    test_graphs = {k: load_split(v) for k, v in SPLITS.items()}
    for k, v in test_graphs.items():
        print(f"  {k}: {len(v)} graphs")

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH)
    test_loaders = {k: DataLoader(v, batch_size=BATCH) for k, v in test_graphs.items()}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VulGNN(
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        embed_dim=EMBED_DIM,
        edge_dim=EDGE_DIM,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nVulGNN parameters: {n_params:,}")

    # Weighted cross-entropy for class imbalance
    weights = compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = 0

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, batch.y.squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        val_f1, val_acc = evaluate(model, valid_loader)
        avg_loss = total_loss / max(n_batches, 1)

        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"val_F1={val_f1:.4f} | val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  *** new best checkpoint (val_F1={val_f1:.4f}) ***")

    print(f"\nBest val F1={best_val_f1:.4f} at epoch {best_epoch}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    results = {}
    for split_name, loader in test_loaders.items():
        f1, acc = evaluate(model, loader)
        results[split_name] = {'f1': round(f1 * 100, 2), 'accuracy': round(acc * 100, 2)}
        print(f"  {split_name}: F1={f1*100:.2f}%  acc={acc*100:.2f}%")

    results['best_val_f1'] = round(best_val_f1 * 100, 2)
    results['best_epoch']  = best_epoch
    results['model'] = 'VulGNN'

    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_OUT}")


if __name__ == '__main__':
    main()
