#!/usr/bin/env python3
"""
Exp 10: REVEAL-NoId — causal ablation for identifier exposure.

Retrains REVEAL with node features stripped of the 100-dim Word2Vec component
(dims 69-168 zeroed out; only 69-dim structural one-hot is kept).

If renaming ASR drops significantly vs. the full-feature REVEAL, this proves
that identifier embeddings CAUSE REVEAL's vulnerability to renaming.

Architecture: Phase 1 GGNN only (no Phase 2 SMOTE/MLP) — sufficient for causal comparison.
Output: devign_full/reveal_noid_seedSEED_results.json
"""

import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parent
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"

# Structural dims only (first 69 of 169) — Word2Vec dims 69-168 are zeroed
STRUCT_DIM   = 69
HIDDEN       = 200
NUM_BLOCKS   = 4
STEPS_PER    = 2
BATCH_SIZE   = 128
LR           = 1e-4
FOCAL_GAMMA  = 3.0
MAX_EPOCHS   = 150
PATIENCE     = 25
SEEDS        = [42, 1337, 7, 100, 999]


# ---------------------------------------------------------------------------
# Data loading — zero the Word2Vec dims
# ---------------------------------------------------------------------------

def load_graphs(json_path: Path) -> list:
    with open(json_path) as f:
        records = json.load(f)
    graphs = []
    for rec in records:
        nf = torch.tensor(rec["node_features"], dtype=torch.float32)
        # Zero out Word2Vec dims (69-168), keep only structural one-hot (0-68)
        nf[:, STRUCT_DIM:] = 0.0
        # Pad 169-dim → 200-dim
        if nf.shape[1] < HIDDEN:
            pad = torch.zeros(nf.shape[0], HIDDEN - nf.shape[1])
            nf = torch.cat([nf, pad], dim=1)
        target = int(rec["targets"][0][0])
        raw_edges = rec.get("graph", [])
        if raw_edges:
            srcs = [e[0] for e in raw_edges]
            dsts = [e[2] for e in raw_edges]
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        y = torch.tensor([float(target)], dtype=torch.float32)
        graphs.append(Data(x=nf, edge_index=edge_index, y=y))
    return graphs


def make_balanced_loader(graphs, batch_size):
    labels = [int(g.y.item()) for g in graphs]
    pos = sum(labels); neg = len(labels) - pos
    w   = [1.0 / neg if l == 0 else 1.0 / pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


# ---------------------------------------------------------------------------
# Model — same ResGGNN as REVEAL faithful
# ---------------------------------------------------------------------------

class ResGGNNBlock(nn.Module):
    def __init__(self, hidden, steps=STEPS_PER):
        super().__init__()
        self.conv = GatedGraphConv(out_channels=hidden, num_layers=steps)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        return self.norm(x + self.conv(x, edge_index))


class RevealGNN(nn.Module):
    def __init__(self, hidden=HIDDEN, num_blocks=NUM_BLOCKS, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([ResGGNNBlock(hidden) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, batch):
        h = x
        for block in self.blocks:
            h = block(h, edge_index)
        h = torch.tanh(h)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.head(g).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss & eval
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy().astype(int)


def best_threshold(logits, labels):
    best_thr, best_f1 = 0.5, 0.0
    probs = 1 / (1 + np.exp(-logits))
    for thr in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def eval_metrics(logits, labels, thr):
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thr).astype(int)
    return {
        "acc": accuracy_score(labels, preds) * 100,
        "pr":  precision_score(labels, preds, zero_division=0) * 100,
        "rc":  recall_score(labels, preds, zero_division=0) * 100,
        "f1":  f1_score(labels, preds, zero_division=0) * 100,
        "n":   len(labels),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_graphs, valid_graphs, device):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH_SIZE, shuffle=False)

    best_state, best_val_f1, best_thr, no_improve = (
        copy.deepcopy(model.state_dict()), 0.0, 0.5, 0
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_b = 0.0, 0
        for batch in make_balanced_loader(train_graphs, BATCH_SIZE):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss   = focal_loss(logits, batch.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item(); n_b += 1
        scheduler.step()

        val_logits, val_labels = predict(model, valid_loader, device)
        thr, val_f1 = best_threshold(val_logits, val_labels)
        val_f1 *= 100

        print(f"Epoch {epoch:3d}  loss={total_loss/max(n_b,1):.4f}  "
              f"val_F1={val_f1:.2f}%  thr={thr:.2f}", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1, best_thr, no_improve = val_f1, thr, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[REVEAL-NoId seed={args.seed}] Device: {device}", flush=True)

    print("Loading graphs (structural features only — Word2Vec zeroed)…", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "train_GGNNinput.json")
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "valid_GGNNinput.json")
    print(f"  train={len(train_graphs)}  valid={len(valid_graphs)}", flush=True)

    model = RevealGNN().to(device)
    model, best_thr = train(model, train_graphs, valid_graphs, device)

    # Evaluate all 8 conditions
    conditions = {
        "clean":    DEVIGN_INPUT / "originals_train"    / "test_GGNNinput.json",
        "ren":      DEVIGN_INPUT / "obf_identifier_test" / "test_GGNNinput.json",
        "dead":     DEVIGN_INPUT / "obf_deadcode_test"   / "test_GGNNinput.json",
        "cf":       DEVIGN_INPUT / "obf_controlflow_test"/ "test_GGNNinput.json",
        "ren_dead": DEVIGN_INPUT / "pairwise_test"       / "obf_ren_dead_test_GGNNinput.json",
        "ren_cf":   DEVIGN_INPUT / "pairwise_test"       / "obf_ren_cf_test_GGNNinput.json",
        "dead_cf":  DEVIGN_INPUT / "pairwise_test"       / "obf_dead_cf_test_GGNNinput.json",
        "compound": DEVIGN_INPUT / "obf_compound_test"   / "test_GGNNinput.json",
    }

    results = {}
    clean_f1 = None

    for cond, path in conditions.items():
        graphs = load_graphs(path)
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        logits, labels = predict(model, loader, device)
        m = eval_metrics(logits, labels, best_thr)
        results[cond] = m
        if clean_f1 is None:
            clean_f1 = m["f1"]
        delta = m["f1"] - clean_f1
        print(f"  {cond:8s}  F1={m['f1']:.2f}%  ΔF1={delta:+.2f}pp", flush=True)

    results["_meta"] = {
        "model":       "reveal_noid",
        "seed":        args.seed,
        "struct_dim":  STRUCT_DIM,
        "total_dim":   169,
        "zeroed_dims": "69-168 (Word2Vec)",
    }

    out = THESIS_ROOT / "devign_full" / f"reveal_noid_seed{args.seed}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}", flush=True)

    # Delta-F1 summary
    print("\n=== REVEAL-NoId Delta-F1 Summary ===", flush=True)
    for cond in ["ren", "dead", "cf", "compound"]:
        delta = results[cond]["f1"] - clean_f1
        print(f"  Δ{cond:8s} = {delta:+.2f}pp", flush=True)


if __name__ == "__main__":
    main()
