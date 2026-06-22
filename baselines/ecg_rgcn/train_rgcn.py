#!/usr/bin/env python3
"""
Relational GCN (R-GCN) vulnerability detector.

Key fix: uses RGCNConv with 5 relation types, giving each edge type
(CFG, data flow, control dep, DOM, POST_DOM) its own weight matrix.
This is structurally equivalent to REVEAL's original per-edge-type
message functions — the part missing from all previous PyG attempts.

Architecture:
  3 × RGCNConv layers (not 8) — fewer steps to reduce oversmoothing
  Each layer: LayerNorm + residual connection
  Readout: cat(mean_pool, max_pool) → 400-dim
  Head: Linear(400,128) → ReLU → Dropout → Linear(128,1)

Training:
  WeightedRandomSampler: 50/50 balanced mini-batches
  Focal loss γ=2.0
  F1-based early stopping (patience=20)
"""

import copy
import json
import sys
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
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"
MODEL_DIR    = SCRIPT_DIR / "models" / "rgcn"
EMBED_DIR    = THESIS_ROOT / "devign_full" / "after_ggnn_rgcn"

# Edge type remapping: raw Joern IDs → 0-indexed
EDGE_REMAP = {3: 0, 6: 1, 7: 2, 9: 3, 10: 4}
NUM_RELATIONS = 5
HIDDEN        = 200
NUM_LAYERS    = 3
BATCH_SIZE    = 128
LR            = 1e-4
FOCAL_GAMMA   = 2.0
MAX_EPOCHS    = 150
PATIENCE      = 20


# ---------------------------------------------------------------------------
# Data
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
            etypes = [EDGE_REMAP.get(e[1], 0) for e in raw]  # remap to 0-4
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            edge_type  = torch.tensor(etypes, dtype=torch.long)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_type  = torch.tensor([0], dtype=torch.long)
        y = torch.tensor([float(target)], dtype=torch.float32)
        graphs.append(Data(x=nf, edge_index=edge_index, edge_type=edge_type, y=y))
    return graphs


def balanced_loader(graphs, batch_size):
    labels = [int(g.y.item()) for g in graphs]
    pos = sum(labels); neg = len(labels) - pos
    w = [1.0/neg if l == 0 else 1.0/pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RGCNBlock(nn.Module):
    """Single R-GCN layer: separate weight matrix per edge type + residual."""
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.conv = RGCNConv(in_channels, out_channels, num_relations,
                             aggr='mean')       # mean agg → scale-invariant
        self.proj = (nn.Linear(in_channels, out_channels)
                     if in_channels != out_channels else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_type):
        return self.norm(self.proj(x) + self.conv(x, edge_index, edge_type))


class RelationalGNN(nn.Module):
    def __init__(self, in_channels, hidden=HIDDEN, num_layers=NUM_LAYERS,
                 num_relations=NUM_RELATIONS, dropout=0.3):
        super().__init__()
        dims = [in_channels] + [hidden] * num_layers
        self.layers = nn.ModuleList([
            RGCNBlock(dims[i], dims[i+1], num_relations)
            for i in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_type, batch):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index, edge_type))
        g = torch.cat([global_mean_pool(h, batch),
                        global_max_pool(h, batch)], dim=1)
        return self.head(g).squeeze(-1)

    def embed(self, x, edge_index, edge_type, batch):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, edge_index, edge_type))
        return torch.cat([global_mean_pool(h, batch),
                           global_max_pool(h, batch)], dim=1)


# ---------------------------------------------------------------------------
# Loss / eval helpers
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for b in loader:
        b = b.to(device)
        logits_all.append(model(b.x, b.edge_index, b.edge_type, b.batch).cpu())
        labels_all.append(b.y.cpu())
    return torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy().astype(int)


def best_f1_threshold(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def eval_metrics(logits, labels, thr):
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thr).astype(int)
    return (accuracy_score(labels, preds) * 100,
            precision_score(labels, preds, zero_division=0) * 100,
            recall_score(labels, preds, zero_division=0) * 100,
            f1_score(labels, preds, zero_division=0) * 100)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_graphs, valid_loader, device):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    best_thr    = 0.5
    no_improve  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loader     = balanced_loader(train_graphs, BATCH_SIZE)
        total_loss = 0.0
        n_batches  = 0
        for b in loader:
            b = b.to(device)
            logits = model(b.x, b.edge_index, b.edge_type, b.batch)
            loss   = focal_loss(logits, b.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        val_logits, val_labels = predict(model, valid_loader, device)
        thr, val_f1 = best_f1_threshold(val_logits, val_labels)
        val_f1 *= 100
        val_rc   = recall_score(val_labels,
                                (1/(1+np.exp(-val_logits)) >= thr).astype(int),
                                zero_division=0) * 100
        val_pr   = precision_score(val_labels,
                                   (1/(1+np.exp(-val_logits)) >= thr).astype(int),
                                   zero_division=0) * 100

        print(f"Epoch {epoch:3d}  loss={total_loss/max(n_batches,1):.4f}  "
              f"val_F1={val_f1:.2f}%  val_Pr={val_pr:.1f}%  val_Rc={val_rc:.1f}%  "
              f"thr={thr:.2f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1
            best_thr    = thr
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_thr, best_val_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}  "
              f"({free/1e9:.1f}/{total/1e9:.1f} GB free)", flush=True)

    print("\nLoading graphs…", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train/train_GGNNinput.json")
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train/valid_GGNNinput.json")
    test_graphs  = load_graphs(DEVIGN_INPUT / "originals_train/test_GGNNinput.json")
    print(f"  train={len(train_graphs)}  valid={len(valid_graphs)}  test={len(test_graphs)}",
          flush=True)

    pos = sum(int(g.y.item()) for g in train_graphs)
    neg = len(train_graphs) - pos
    print(f"  Train: pos={pos}  neg={neg}  ({100*pos/len(train_graphs):.1f}% positive)",
          flush=True)

    # Check edge type remap coverage
    all_raw = set()
    for g in train_graphs[:100]:
        if hasattr(g, 'edge_type'):
            all_raw.update(g.edge_type.tolist())
    print(f"  Remapped edge types (0-indexed): {sorted(all_raw)}", flush=True)

    in_dim = train_graphs[0].x.shape[1]
    print(f"  Input dim: {in_dim}", flush=True)

    valid_loader = DataLoader(valid_graphs, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=128, shuffle=False)

    model = RelationalGNN(in_channels=in_dim, hidden=HIDDEN,
                          num_layers=NUM_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: RelationalGNN  {NUM_LAYERS} R-GCN layers  "
          f"{NUM_RELATIONS} relations  hidden={HIDDEN}  params={n_params:,}", flush=True)

    model, best_thr, best_val_f1 = train(model, train_graphs, valid_loader, device)

    test_logits, test_labels = predict(model, test_loader, device)
    acc, pr, rc, f1 = eval_metrics(test_logits, test_labels, best_thr)
    print(f"\nTest  Acc={acc:.2f}%  Pr={pr:.2f}%  Rc={rc:.2f}%  F1={f1:.2f}%  "
          f"thr={best_thr:.2f}", flush=True)

    degenerate = rc > 95.0 or f1 <= 63.7
    print(f"\n{'DEGENERATE' if degenerate else 'NON-DEGENERATE'}: "
          f"F1={f1:.2f}%  Rc={rc:.2f}%", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'best_thr': best_thr,
                'val_f1': best_val_f1, 'test': {'acc': acc, 'pr': pr, 'rc': rc, 'f1': f1},
                'in_dim': in_dim},
               MODEL_DIR / 'best.pt')

    # Extract and save embeddings for robustness eval
    if not degenerate:
        print("\nExtracting embeddings for robustness eval…", flush=True)
        EMBED_DIR.mkdir(parents=True, exist_ok=True)

        def extract(graphs, fname):
            model.eval()
            loader = DataLoader(graphs, batch_size=128, shuffle=False)
            embs, labs = [], []
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    embs.append(model.embed(b.x, b.edge_index, b.edge_type, b.batch).cpu().numpy())
                    labs.extend(b.y.cpu().numpy().astype(int).tolist())
            embs = np.vstack(embs)
            out = [{'graph_feature': embs[i].tolist(), 'target': labs[i]}
                   for i in range(len(labs))]
            with open(EMBED_DIR / fname, 'w') as f:
                json.dump(out, f)
            print(f"  Saved {len(out)} embeddings → {EMBED_DIR/fname}", flush=True)

        extract(train_graphs, 'train_GGNNinput_graph.json')
        extract(valid_graphs, 'valid_GGNNinput_graph.json')
        extract(test_graphs,  'test_GGNNinput_graph.json')

        for name, path in [
            ('obf_identifier',  DEVIGN_INPUT / "obf_identifier_test/test_GGNNinput.json"),
            ('obf_deadcode',    DEVIGN_INPUT / "obf_deadcode_test/test_GGNNinput.json"),
            ('obf_controlflow', DEVIGN_INPUT / "obf_controlflow_test/test_GGNNinput.json"),
        ]:
            obf_graphs = load_graphs(path)
            extract(obf_graphs, f'{name}_test_GGNNinput_graph.json')

        print("\nRunning REVEAL robustness eval on R-GCN embeddings…", flush=True)
        import subprocess
        eval_script = (THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe/representation_learning"
                       / "reveal_robustness_eval.py")
        subprocess.run([sys.executable, "-u", str(eval_script),
                        "--after_ggnn_dir", str(EMBED_DIR),
                        "--output_json", str(EMBED_DIR / "robustness_results.json")],
                       check=True)
    else:
        print("\nModel degenerate — skipping robustness eval.", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
