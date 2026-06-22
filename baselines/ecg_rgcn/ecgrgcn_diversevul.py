"""
ecgrgcn_diversevul.py — ECG RGCN (Relational GCN) on DiverseVul (balanced subset).

Uses 2-edge CPG format (AST=0, CFG=1) instead of 5-edge GGNN format.
Architecture: 3-layer R-GCN + global mean+max pool + MLP head.

Usage:
    CUDA_VISIBLE_DEVICES=X python ~/ecgrgcn_diversevul.py

Outputs:
    ~/thesis/devign_full/diversevul_ecgrgcn_multiseed_results.json
"""

import json, os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, os.path.expanduser("~"))
from diversevul_cpg_parser import load_split, NUM_NODE_TYPES, NUM_EDGE_TYPES

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 100
LR         = 1e-3
BATCH      = 128
HIDDEN     = 128
NUM_LAYERS = 3
EMBED_DIM  = 32
DROPOUT    = 0.1
FOCAL_GAMMA= 2.0
PATIENCE   = 20

RESULTS_DIR = os.path.expanduser('~/thesis/devign_full')
CKPT_DIR    = os.path.expanduser('~/ecgrgcn_diversevul_ckpts')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 1337, 7, 100, 999]
SPLITS = {
    'test': 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class ECGRGCNDiverseVul(nn.Module):
    """R-GCN with 2 relation types (AST, CFG) for DiverseVul CPG data."""
    def __init__(self, num_node_types, embed_dim, hidden, num_layers, dropout):
        super().__init__()
        self.embed      = nn.Embedding(num_node_types + 1, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.convs      = nn.ModuleList([
            RGCNConv(hidden, hidden, num_relations=NUM_EDGE_TYPES)
            for _ in range(num_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dim() > 1: x = x.squeeze(-1)
        x = F.relu(self.input_proj(self.embed(x.long())))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x   = conv(x, edge_index, edge_type)
            x   = norm(x + res)
            x   = F.relu(x)
            x   = self.dropout(x)
        mean_p = global_mean_pool(x, batch)
        max_p  = global_max_pool(x, batch)
        return self.classifier(torch.cat([mean_p, max_p], dim=-1))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, loader):
    model.eval(); preds, truths = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            logits = model(b)
            preds.extend(logits.argmax(-1).cpu().tolist())
            truths.extend(b.y.long().cpu().tolist())
    return (f1_score(truths, preds, zero_division=0),
            accuracy_score(truths, preds),
            precision_score(truths, preds, zero_division=0),
            recall_score(truths, preds, zero_division=0))


def train_one_seed(seed, train_graphs, valid_graphs, test_graphs):
    set_seed(seed)
    print(f"\n{'='*55}\n  ECG RGCN DiverseVul  Seed {seed}\n{'='*55}")
    ckpt = os.path.join(CKPT_DIR, f'ecgrgcn_dv_seed{seed}.pt')

    labels = [int(g.y.item()) for g in train_graphs]
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    w = [1.0/n_neg if l == 0 else 1.0/n_pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_graphs, batch_size=BATCH, sampler=sampler)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH)
    test_loaders = {k: DataLoader(v, batch_size=BATCH) for k, v in test_graphs.items()}

    model     = ECGRGCNDiverseVul(NUM_NODE_TYPES, EMBED_DIM, HIDDEN, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0; patience_cnt = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0
        for b in train_loader:
            b = b.to(DEVICE); optimizer.zero_grad()
            logits = model(b); loss = criterion(logits, b.y.long().squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()
        scheduler.step()
        val_f1, _, _, _ = evaluate(model, valid_loader)
        print(f"  Ep {epoch:3d}/{EPOCHS} loss={total_loss/max(len(train_loader),1):.4f} val_F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; patience_cnt = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: print(f"  Early stop ep {epoch}"); break

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    results = {'seed': seed, 'best_val_f1': round(best_val_f1*100, 2)}
    for split_name, loader in test_loaders.items():
        f1, acc, pr, rc = evaluate(model, loader)
        results[split_name] = {'f1': round(f1*100,2), 'acc': round(acc*100,2),
                                'pr': round(pr*100,2), 'rc': round(rc*100,2)}
        print(f"  {split_name}: F1={f1*100:.2f}%")

    out = os.path.join(RESULTS_DIR, f'diversevul_ecgrgcn_seed{seed}_results.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    return results


def aggregate(all_results):
    conditions = list(SPLITS.keys())
    agg = {}
    for cond in conditions:
        f1s = [r[cond]['f1'] for r in all_results if cond in r]
        agg[cond] = {'f1_mean': round(float(np.mean(f1s)),2),
                     'f1_std':  round(float(np.std(f1s)),2), 'all_f1': f1s}
    base = agg['test']['f1_mean']
    for cond in conditions[1:]:
        agg[cond]['delta_f1'] = round(agg[cond]['f1_mean'] - base, 2)
    agg.update({'n_seeds': len(all_results), 'seeds': [r['seed'] for r in all_results],
                'model': 'ECG RGCN', 'dataset': 'DiverseVul'})
    return agg


def main():
    print(f"Device: {DEVICE}")
    print("Loading DiverseVul CPG splits...")
    t0 = time.time()
    train_graphs = load_split('train')
    valid_graphs = load_split('valid')
    test_graphs  = {k: load_split(v) for k, v in SPLITS.items()}
    print(f"Loaded in {time.time()-t0:.1f}s: train={len(train_graphs)}, valid={len(valid_graphs)}")

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, train_graphs, valid_graphs, test_graphs)
        all_results.append(r)

    agg = aggregate(all_results)
    out = os.path.join(RESULTS_DIR, 'diversevul_ecgrgcn_multiseed_results.json')
    with open(out, 'w') as f: json.dump(agg, f, indent=2)
    print(f"\nResults → {out}")
    for cond, m in agg.items():
        if isinstance(m, dict):
            delta = f"  ΔF1={m.get('delta_f1',0):+.2f}" if cond != 'test' else ''
            print(f"  {cond:<30} F1={m['f1_mean']:.2f}±{m['f1_std']:.2f}%{delta}")


if __name__ == '__main__':
    main()
