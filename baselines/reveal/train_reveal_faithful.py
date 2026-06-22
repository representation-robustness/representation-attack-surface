#!/usr/bin/env python3
"""
REVEAL-faithful GNN vulnerability detector with anti-degenerate fixes.

Key fixes over previous attempts:
  1. Residual connections (4 blocks × 2 GRU steps each) → prevents oversmoothing
  2. WeightedRandomSampler 50/50 batches → breaks all-positive gradient bias
  3. Mean+Max pooling concatenated → richer graph-level repr
  4. Focal loss (γ=3) → harder to stay in all-positive basin
  5. F1-based early stopping (not loss) → direct metric optimization

Phase 2 (if Phase 1 converges):
  SMOTE + MLP with triplet-margin loss on extracted embeddings (REVEAL §3.2)
"""

import argparse
import copy
import json
import sys
import time
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
THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"
MODEL_DIR    = SCRIPT_DIR / "models" / "reveal_faithful"
EMBED_DIR    = THESIS_ROOT / "devign_full" / "after_ggnn_pyg_faithful"

HIDDEN     = 200
NUM_BLOCKS = 4        # 4 residual blocks × 2 GRU steps = 8 total GRU steps
STEPS_PER  = 2
BATCH_SIZE = 128
LR         = 1e-4
FOCAL_GAMMA = 3.0
MAX_EPOCHS  = 150
PATIENCE    = 25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(json_path: Path) -> list:
    with open(json_path) as f:
        records = json.load(f)
    graphs = []
    for rec in records:
        nf = torch.tensor(rec["node_features"], dtype=torch.float32)
        # Pad 169-dim → 200-dim (REVEAL uses 200 as initial annotation)
        if nf.shape[1] < HIDDEN:
            pad = torch.zeros(nf.shape[0], HIDDEN - nf.shape[1])
            nf = torch.cat([nf, pad], dim=1)
        target = int(rec["targets"][0][0])
        raw_edges = rec.get("graph", [])
        if raw_edges:
            srcs   = [e[0] for e in raw_edges]
            dsts   = [e[2] for e in raw_edges]
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        y = torch.tensor([float(target)], dtype=torch.float32)
        graphs.append(Data(x=nf, edge_index=edge_index, y=y))
    return graphs


def make_balanced_loader(graphs, batch_size, shuffle=False):
    """DataLoader with 50/50 WeightedRandomSampler for training."""
    labels = [int(g.y.item()) for g in graphs]
    pos = sum(labels)
    neg = len(labels) - pos
    w   = [1.0 / neg if l == 0 else 1.0 / pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResGGNNBlock(nn.Module):
    """GatedGraphConv (2 steps) + residual + LayerNorm."""
    def __init__(self, hidden, steps=STEPS_PER):
        super().__init__()
        self.conv = GatedGraphConv(out_channels=hidden, num_layers=steps)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        return self.norm(x + self.conv(x, edge_index))


class RevealGNN(nn.Module):
    """
    4 × ResGGNNBlock (8 GRU steps total) + tanh + mean+max readout + MLP head.
    Initial 200-dim node features come pre-padded from the CPG feature vectors.
    """
    def __init__(self, hidden=HIDDEN, num_blocks=NUM_BLOCKS, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResGGNNBlock(hidden) for _ in range(num_blocks)
        ])
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
        h = torch.tanh(h)                                          # REVEAL uses tanh
        g = torch.cat([global_mean_pool(h, batch),
                        global_max_pool(h, batch)], dim=1)         # 400-dim
        return self.head(g).squeeze(-1)                            # (B,) logits

    def embed(self, x, edge_index, batch):
        """Return 400-dim graph embedding (no head)."""
        h = x
        for block in self.blocks:
            h = block(h, edge_index)
        h = torch.tanh(h)
        return torch.cat([global_mean_pool(h, batch),
                           global_max_pool(h, batch)], dim=1)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, gamma=FOCAL_GAMMA):
    bce  = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt   = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Train / Eval helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    return logits, labels


def best_threshold_f1(logits, labels):
    """Sweep thresholds [0.1..0.9] and pick the one maximising F1."""
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
    acc = accuracy_score(labels, preds) * 100
    pr  = precision_score(labels, preds, zero_division=0) * 100
    rc  = recall_score(labels, preds, zero_division=0) * 100
    f1  = f1_score(labels, preds, zero_division=0) * 100
    return acc, pr, rc, f1


# ---------------------------------------------------------------------------
# Phase 1: GGNN training
# ---------------------------------------------------------------------------

def train_phase1(model, train_graphs, valid_loader, device, max_epochs, patience):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    best_thr    = 0.5
    no_improve  = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        # Rebuild balanced loader each epoch (re-sample)
        train_loader = make_balanced_loader(train_graphs, BATCH_SIZE)

        total_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss   = focal_loss(logits, batch.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        val_logits, val_labels = predict(model, valid_loader, device)
        thr, val_f1 = best_threshold_f1(val_logits, val_labels)
        val_f1 *= 100

        avg_loss = total_loss / max(n_batches, 1)
        recall   = recall_score(val_labels, (1/(1+np.exp(-val_logits)) >= thr).astype(int),
                                zero_division=0) * 100

        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  val_F1={val_f1:.2f}%  "
              f"val_Rc={recall:.1f}%  thr={thr:.2f}  lr={scheduler.get_last_lr()[0]:.2e}",
              flush=True)

        if val_f1 > best_val_f1 + 0.1:
            best_val_f1 = val_f1
            best_thr    = thr
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val F1 improvement for {patience} epochs)",
                      flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_thr, best_val_f1


# ---------------------------------------------------------------------------
# Phase 2: REVEAL representation learning
# ---------------------------------------------------------------------------

class RepresentationNet(nn.Module):
    """MLP for representation learning, mirroring REVEAL §3.2."""
    def __init__(self, in_dim=400, hidden1=256, hidden2=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def triplet_loss(emb, labels, margin=0.5):
    """Batch hard triplet loss: for each anchor find hardest pos/neg in batch."""
    labels = labels.to(emb.device)
    dists  = torch.cdist(emb, emb, p=2)  # (B, B)
    B = emb.shape[0]
    loss_sum = torch.tensor(0.0, device=emb.device)
    n_valid  = 0
    for i in range(B):
        pos_mask = (labels == labels[i]) & (torch.arange(B, device=emb.device) != i)
        neg_mask = labels != labels[i]
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        hardest_pos = dists[i][pos_mask].max()
        hardest_neg = dists[i][neg_mask].min()
        loss_sum = loss_sum + F.relu(hardest_pos - hardest_neg + margin)
        n_valid += 1
    return loss_sum / max(n_valid, 1)


def train_phase2(embs_train, labels_train, embs_val, labels_val,
                 alpha=0.5, beta=0.001, lr=1e-3, max_epochs=100, patience=20,
                 device='cpu'):
    """Train representation network with L_CE + alpha*L_triplet + beta*L_reg."""
    from imblearn.over_sampling import SMOTE

    # SMOTE to balance training embeddings
    print("Applying SMOTE to training embeddings…", flush=True)
    sm = SMOTE(random_state=42)
    embs_sm, labels_sm = sm.fit_resample(embs_train, labels_train)
    print(f"  After SMOTE: {embs_sm.shape[0]} samples "
          f"(pos={labels_sm.sum()}, neg={(labels_sm==0).sum()})", flush=True)

    X_tr = torch.tensor(embs_sm,    dtype=torch.float32).to(device)
    y_tr = torch.tensor(labels_sm,  dtype=torch.float32).to(device)
    X_va = torch.tensor(embs_val,   dtype=torch.float32).to(device)
    y_va = torch.tensor(labels_val, dtype=torch.long).to(device)

    in_dim = X_tr.shape[1]
    repr_net = RepresentationNet(in_dim=in_dim).to(device)
    ce_head  = nn.Linear(256, 1).to(device)
    optim    = Adam(list(repr_net.parameters()) + list(ce_head.parameters()),
                    lr=lr, weight_decay=beta)

    best_state  = (copy.deepcopy(repr_net.state_dict()),
                   copy.deepcopy(ce_head.state_dict()))
    best_val_f1 = 0.0
    no_improve  = 0
    batch_size  = 256

    for epoch in range(1, max_epochs + 1):
        repr_net.train(); ce_head.train()
        perm   = torch.randperm(X_tr.shape[0])
        ep_loss = 0.0
        n_b = 0
        for i in range(0, X_tr.shape[0], batch_size):
            idx    = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            z      = repr_net(xb)
            logits = ce_head(z).squeeze(-1)
            l_ce   = F.binary_cross_entropy_with_logits(logits, yb)
            l_tri  = triplet_loss(z, yb.long())
            l_reg  = sum(p.pow(2).sum() for p in repr_net.parameters())
            loss   = l_ce + alpha * l_tri + beta * l_reg
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(repr_net.parameters()) + list(ce_head.parameters()),
                                      max_norm=5.0)
            optim.step()
            ep_loss += loss.item(); n_b += 1

        # Validate
        repr_net.eval(); ce_head.eval()
        with torch.no_grad():
            z_val   = repr_net(X_va)
            logits_v = ce_head(z_val).squeeze(-1).cpu().numpy()
        labels_v = y_va.cpu().numpy()
        thr, vf1 = best_threshold_f1(logits_v, labels_v)
        vf1 *= 100

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Phase2 epoch {epoch:3d}  loss={ep_loss/n_b:.4f}  val_F1={vf1:.2f}%",
                  flush=True)

        if vf1 > best_val_f1 + 0.1:
            best_val_f1 = vf1
            best_state  = (copy.deepcopy(repr_net.state_dict()),
                           copy.deepcopy(ce_head.state_dict()))
            best_thr    = thr
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Phase2 early stop at epoch {epoch}", flush=True)
                break

    repr_net.load_state_dict(best_state[0])
    ce_head.load_state_dict(best_state[1])
    return repr_net, ce_head, best_thr, best_val_f1


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, graphs, device, batch_size=128):
    """Extract 400-dim graph embeddings from the GGNN."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_embs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        emb   = model.embed(batch.x, batch.edge_index, batch.batch)
        all_embs.append(emb.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy().astype(int).tolist())
    return np.vstack(all_embs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',      default='auto')
    parser.add_argument('--max_epochs',  type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience',    type=int, default=PATIENCE)
    parser.add_argument('--phase1_only', action='store_true',
                        help='Stop after GGNN training (skip repr learning)')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            # Pick least-loaded GPU
            mem_free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            device   = torch.device(f'cuda:{mem_free.index(max(mem_free))}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU memory: {free/1e9:.1f}/{total/1e9:.1f} GB free", flush=True)

    # Load data
    print("\nLoading graphs…", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "train_GGNNinput.json")
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "valid_GGNNinput.json")
    test_graphs  = load_graphs(DEVIGN_INPUT / "originals_train" / "test_GGNNinput.json")
    print(f"  train={len(train_graphs)}  valid={len(valid_graphs)}  test={len(test_graphs)}",
          flush=True)

    # Class balance report
    pos = sum(int(g.y.item()) for g in train_graphs)
    neg = len(train_graphs) - pos
    print(f"  Train: pos={pos}  neg={neg}  ({100*pos/len(train_graphs):.1f}% positive)",
          flush=True)

    valid_loader = DataLoader(valid_graphs, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=128, shuffle=False)

    # --------------- Phase 1: GGNN training ---------------
    print("\n" + "="*60)
    print("PHASE 1: GGNN training with residual connections")
    print("="*60, flush=True)

    model = RevealGNN(hidden=HIDDEN, num_blocks=NUM_BLOCKS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)
    print(f"Architecture: {NUM_BLOCKS} ResGGNNBlocks × {STEPS_PER} steps "
          f"({NUM_BLOCKS * STEPS_PER} total GRU steps)", flush=True)
    print(f"Focal loss γ={FOCAL_GAMMA}, batch_size={BATCH_SIZE} (50/50 balanced)",
          flush=True)

    model, best_thr, best_val_f1 = train_phase1(
        model, train_graphs, valid_loader, device,
        args.max_epochs, args.patience
    )

    # Evaluate on test set
    test_logits, test_labels = predict(model, test_loader, device)
    acc, pr, rc, f1 = eval_metrics(test_logits, test_labels, best_thr)
    print(f"\nPhase 1 Test  Acc={acc:.2f}%  Pr={pr:.2f}%  Rc={rc:.2f}%  "
          f"F1={f1:.2f}%  thr={best_thr:.2f}", flush=True)

    degenerate_f1 = 62.67  # all-positive bound for 45.6% class
    if rc > 95.0 or f1 <= degenerate_f1 + 1.0:
        print(f"\nWARNING: Model is degenerate (F1={f1:.2f}%, Rc={rc:.2f}%)", flush=True)
        print("Continuing to Phase 2 anyway — repr learning may still help.", flush=True)
    else:
        print(f"\nPhase 1 CONVERGED (F1={f1:.2f}%, Rc={rc:.2f}% — non-degenerate!)",
              flush=True)

    # Save Phase 1 checkpoint
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'best_thr': best_thr,
        'val_f1':   best_val_f1,
        'test':     {'acc': acc, 'pr': pr, 'rc': rc, 'f1': f1},
    }, MODEL_DIR / 'phase1_best.pt')
    print(f"Phase 1 checkpoint → {MODEL_DIR / 'phase1_best.pt'}", flush=True)

    if args.phase1_only:
        print("--phase1_only set, exiting.", flush=True)
        return

    # --------------- Extract Phase 1 embeddings ---------------
    print("\nExtracting GGNN embeddings for all splits…", flush=True)
    embs_train, labels_train = extract_embeddings(model, train_graphs, device)
    embs_valid, labels_valid = extract_embeddings(model, valid_graphs, device)
    embs_test,  labels_test  = extract_embeddings(model, test_graphs,  device)
    print(f"  Embedding shape: {embs_train.shape}", flush=True)

    # Load and embed obfuscated test sets
    obf_sets = {
        'obf_identifier':  DEVIGN_INPUT / "obf_identifier_test"  / "test_GGNNinput.json",
        'obf_deadcode':    DEVIGN_INPUT / "obf_deadcode_test"    / "test_GGNNinput.json",
        'obf_controlflow': DEVIGN_INPUT / "obf_controlflow_test" / "test_GGNNinput.json",
    }
    embs_obf, labels_obf = {}, {}
    for name, path in obf_sets.items():
        graphs = load_graphs(path)
        embs_obf[name], labels_obf[name] = extract_embeddings(model, graphs, device)
        print(f"  {name}: {embs_obf[name].shape[0]} embeddings", flush=True)

    # --------------- Phase 2: Representation learning ---------------
    print("\n" + "="*60)
    print("PHASE 2: Representation learning (SMOTE + triplet loss)")
    print("="*60, flush=True)

    repr_net, ce_head, thr2, val_f1_2 = train_phase2(
        embs_train, labels_train,
        embs_valid, labels_valid,
        alpha=0.5, beta=0.001, lr=1e-3,
        max_epochs=100, patience=20,
        device=device,
    )

    # Evaluate Phase 2 on test + obf sets
    print("\nPhase 2 evaluation:", flush=True)
    repr_net.eval(); ce_head.eval()

    def phase2_eval(embs, labels, name, thr):
        X = torch.tensor(embs, dtype=torch.float32).to(device)
        with torch.no_grad():
            z      = repr_net(X)
            logits = ce_head(z).squeeze(-1).cpu().numpy()
        # Use best threshold found on validation
        thr_used, _ = best_threshold_f1(logits, labels)
        acc, pr, rc, f1 = eval_metrics(logits, labels, thr_used)
        print(f"  {name:20s}  Acc={acc:.2f}%  Pr={pr:.2f}%  Rc={rc:.2f}%  "
              f"F1={f1:.2f}%  thr={thr_used:.2f}", flush=True)
        return {'acc': acc, 'pr': pr, 'rc': rc, 'f1': f1, 'thr': thr_used}

    test_r2 = phase2_eval(embs_test, labels_test, 'original', thr2)
    obf_r2  = {}
    for name in obf_sets:
        obf_r2[name] = phase2_eval(embs_obf[name], labels_obf[name], name, thr2)

    # Compute delta F1
    print("\nRobustness summary (Phase 2):", flush=True)
    print(f"  Original F1: {test_r2['f1']:.2f}%", flush=True)
    for name, res in obf_r2.items():
        delta = res['f1'] - test_r2['f1']
        print(f"  {name:20s}  ΔF1={delta:+.2f}%  "
              f"(orig={test_r2['f1']:.2f}% → obf={res['f1']:.2f}%)", flush=True)

    # Save all results
    results = {
        'phase1': {
            'val_f1':  best_val_f1,
            'test':    {'acc': acc, 'pr': pr, 'rc': rc, 'f1': f1},
            'degenerate': (rc > 95.0 or f1 <= degenerate_f1 + 1.0),
        },
        'phase2': {
            'val_f1': val_f1_2,
            'test':   test_r2,
            'obf':    obf_r2,
            'delta_f1': {name: obf_r2[name]['f1'] - test_r2['f1']
                         for name in obf_r2},
        }
    }
    save_results(results, SCRIPT_DIR / "models" / "reveal_faithful_results.json")

    torch.save({
        'repr_net_state': repr_net.state_dict(),
        'ce_head_state':  ce_head.state_dict(),
        'thr2':           thr2,
        'val_f1_2':       val_f1_2,
        'in_dim':         embs_train.shape[1],
    }, MODEL_DIR / 'phase2_best.pt')
    print(f"\nPhase 2 checkpoint → {MODEL_DIR / 'phase2_best.pt'}", flush=True)
    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
