#!/usr/bin/env python3
"""
VulGNN adaptation for devign_full CPG features.

Architecture follows Farmer et al. (2026) "Software Vulnerability Detection
Using a Lightweight Graph Neural Network" exactly, except:
  - Node features: 169-dim CPG float vectors (Word2Vec + node-type one-hot)
    instead of StarCoder BPE token sequences
  - Edge features: 4-dim embedding of the 5 GGNN edge types (CFG, data-flow,
    control-dep, DOM, POST-DOM) instead of token-sequence embeddings

Everything else matches the paper:
  - 6 x ConvGroup blocks: GeneralConv (dot-product attn, mean agg) + PReLU
    + GraphNorm + Dropout(0.08)
  - First block: 169 -> 128; subsequent blocks: 128 -> 128
  - Readout: global mean pool
  - Head: Linear(128,128) -> sigmoid -> Linear(128,2)
  - Loss: class-weighted BCE (paper Eq. 7-9)
  - Optimizer: Adam lr=1e-3, beta1=0.9, beta2=0.999, 25 epochs, batch 400
  - Best checkpoint by val F1
  - Additional: WeightedRandomSampler for 50/50 batches (needed on devign_full
    to escape all-positive collapse, as established by ECG-RGCN experiments)
"""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GeneralConv, GraphNorm, global_mean_pool
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"
MODEL_DIR    = SCRIPT_DIR / "models" / "vulgnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EDGE_REMAP     = {3: 0, 6: 1, 7: 2, 9: 3, 10: 4}
NUM_EDGE_TYPES = 5
EDGE_EMB_DIM   = 4      # paper uses de=4 for edge type embeddings

IN_DIM     = 169        # CPG node feature dimension
HIDDEN     = 128        # D in the paper
NUM_LAYERS = 6          # paper default
DROPOUT    = 0.08       # paper value
BATCH_SIZE = 400        # paper value
LR         = 1e-4       # reduced from paper's 1e-3 (1e-3 collapses on devign_full)
FOCAL_GAMMA = 2.0       # focal loss exponent (replaces class-weighted CE)
MAX_EPOCHS = 60         # longer to allow convergence at lower lr
PATIENCE   = 15         # early stopping


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(json_path: Path) -> list:
    with open(json_path) as f:
        records = json.load(f)
    graphs = []
    for rec in records:
        nf     = torch.tensor(rec["node_features"], dtype=torch.float32)
        target = int(rec["targets"][0][0])
        raw    = rec.get("graph", [])
        if raw:
            srcs   = [e[0] for e in raw]
            dsts   = [e[2] for e in raw]
            etypes = [EDGE_REMAP.get(e[1], 0) for e in raw]
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            edge_type  = torch.tensor(etypes, dtype=torch.long)
        else:
            # isolated node: self-loop placeholder
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_type  = torch.tensor([0], dtype=torch.long)
        y = torch.tensor([float(target)], dtype=torch.float32)
        graphs.append(Data(x=nf, edge_index=edge_index,
                           edge_type=edge_type, y=y))
    return graphs


def balanced_loader(graphs, batch_size):
    labels = [int(g.y.item()) for g in graphs]
    pos = sum(labels); neg = len(labels) - pos
    w   = [1.0/neg if l == 0 else 1.0/pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


def plain_loader(graphs, batch_size):
    return DataLoader(graphs, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvGroup(nn.Module):
    """Single VulGNN convolutional block (paper Section III-D)."""
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.conv = GeneralConv(
            in_channels, out_channels,
            in_edge_channels=edge_dim,
            aggr='mean',
            attention=True,
            attention_type='dot_product',
        )
        self.act  = nn.PReLU()
        self.norm = GraphNorm(out_channels)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.conv(x, edge_index, edge_attr)
        h = self.act(h)
        h = self.norm(h, batch)
        h = self.drop(h)
        return h


class VulGNN(nn.Module):
    """VulGNN with CPG float-vector node features."""
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, num_edge_types=NUM_EDGE_TYPES,
                 edge_emb_dim=EDGE_EMB_DIM):
        super().__init__()
        # Input projection: map CPG features to hidden dim
        self.input_proj = nn.Linear(in_dim, hidden)
        # Edge type embedding (paper uses de=4)
        self.edge_emb = nn.Embedding(num_edge_types, edge_emb_dim)
        # Stacked ConvGroup blocks
        self.blocks = nn.ModuleList([
            ConvGroup(hidden, hidden, edge_emb_dim)
            for _ in range(num_layers)
        ])
        # Classification head (paper Eq. 3)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index, edge_type, batch):
        h = self.input_proj(x)
        ea = self.edge_emb(edge_type)
        for block in self.blocks:
            h = block(h, edge_index, ea, batch)
        g = global_mean_pool(h, batch)
        return self.head(g)   # [B, 2] logits


# ---------------------------------------------------------------------------
# Loss: focal loss (replaces paper's class-weighted BCE)
# Focal loss is necessary on devign_full to escape the all-positive minimum,
# as established by ECG-RGCN experiments.
# ---------------------------------------------------------------------------

def focal_loss(logits, labels_float, gamma=FOCAL_GAMMA):
    """Focal loss using the positive-class logit (binary)."""
    # logits shape: [B, 2]; use logit[:,1] - logit[:,0] as binary score
    binary_logit = logits[:, 1] - logits[:, 0]
    bce = F.binary_cross_entropy_with_logits(binary_logit, labels_float, reduction='none')
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.edge_type, b.batch)
        logits_all.append(out.cpu())
        labels_all.append(b.y.cpu())
    logits = torch.cat(logits_all).numpy()   # [N, 2]
    labels = torch.cat(labels_all).numpy().astype(int).flatten()
    # Convert to probabilities then threshold at 0.5 (argmax)
    preds = logits.argmax(axis=1)
    return preds, labels


def eval_metrics(preds, labels):
    return (
        accuracy_score(labels, preds) * 100,
        precision_score(labels, preds, zero_division=0) * 100,
        recall_score(labels, preds, zero_division=0) * 100,
        f1_score(labels, preds, zero_division=0) * 100,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_graphs, valid_graphs, device):
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    no_improve  = 0

    val_loader = plain_loader(valid_graphs, BATCH_SIZE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loader     = balanced_loader(train_graphs, BATCH_SIZE)
        total_loss = 0.0
        n_batches  = 0

        for b in loader:
            b = b.to(device)
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.edge_type, b.batch)
            loss = focal_loss(out, b.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        # Validation
        val_preds, val_labels = predict(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0) * 100
        val_rc = recall_score(val_labels, val_preds, zero_division=0) * 100

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | loss {total_loss/n_batches:.4f} "
              f"| val_F1 {val_f1:.2f}% | val_Rc {val_rc:.2f}%",
              flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
            torch.save(best_state, MODEL_DIR / "best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------

def eval_condition(model, json_path: Path, device):
    graphs = load_graphs(json_path)
    loader = plain_loader(graphs, BATCH_SIZE)
    preds, labels = predict(model, loader, device)
    acc, pr, rc, f1 = eval_metrics(preds, labels)
    return {"acc": acc, "pr": pr, "rc": rc, "f1": f1, "n": len(graphs)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading graphs...", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "train_GGNNinput_graph.json")
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "valid_GGNNinput_graph.json")
    print(f"Train: {len(train_graphs)} | Val: {len(valid_graphs)}", flush=True)

    # Build model
    model = VulGNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    # Train
    model = train(model, train_graphs, valid_graphs, device)

    # Evaluate all conditions
    conditions = {
        "originals":   DEVIGN_INPUT / "originals_train"    / "test_GGNNinput_graph.json",
        "identifier":  DEVIGN_INPUT / "obf_identifier_test" / "test_GGNNinput_graph.json",
        "deadcode":    DEVIGN_INPUT / "obf_deadcode_test"   / "test_GGNNinput_graph.json",
        "controlflow": DEVIGN_INPUT / "obf_controlflow_test"/ "test_GGNNinput_graph.json",
    }

    print("\n=== Robustness Results ===")
    results = {}
    base_f1 = None
    for name, path in conditions.items():
        r = eval_condition(model, path, device)
        results[name] = r
        delta = (r['f1'] - base_f1) if base_f1 is not None else 0.0
        if base_f1 is None:
            base_f1 = r['f1']
        delta_str = f"  Δ={delta:+.2f}%" if name != "originals" else ""
        print(f"{name:15s} Acc={r['acc']:.2f}%  Pr={r['pr']:.2f}%  "
              f"Rc={r['rc']:.2f}%  F1={r['f1']:.2f}%{delta_str}")

    # Save
    out_file = THESIS_ROOT / "devign_full" / "vulgnn_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
