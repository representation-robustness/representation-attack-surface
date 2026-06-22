#!/usr/bin/env python3
"""
Figure 1: ROC Under Identifier Renaming — ECG RGCN, REVEAL, ReGVD
across Devign, Big-Vul, and DiverseVul.

Clean vs. renamed ROC for structure-aware and token-aware GNN models.
"""

import sys, json, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from torch_geometric.loader import DataLoader as PyGLoader

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME        = Path.home()
THESIS      = HOME / "thesis"
DEVIGN_IN   = THESIS / "devign_full" / "devign_input"
OUT_DIR     = THESIS / "figures" / "roc_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH       = 128

print(f"Device: {DEVICE}", flush=True)

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

@torch.no_grad()
def probs_from_pyg(model, loader, binary=False):
    """Return (probs, labels). binary=True → sigmoid on scalar logit."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        if binary:
            p = torch.sigmoid(out.squeeze(-1)).cpu().numpy()
        else:
            p = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(p.tolist())
        all_labels.extend(batch.y.long().cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def roc_from_probs(probs, labels, name, condition):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        print(f"  WARNING: {name} {condition} has single class — skipping", flush=True)
        return None
    fpr, tpr, _ = roc_curve(labels, probs)
    a = sk_auc(fpr, tpr)
    return fpr, tpr, a


# ──────────────────────────────────────────────
# DEVIGN — ECG RGCN
# ──────────────────────────────────────────────

def devign_ecg_rgcn():
    sys.path.insert(0, str(THESIS / "baselines" / "pyg_gnn"))
    from train_rgcn import load_graphs, RelationalGNN, HIDDEN, NUM_LAYERS, NUM_RELATIONS, BATCH_SIZE
    sys.path.pop(0)

    ckpt_path = THESIS / "baselines/pyg_gnn/models/rgcn/best.pt"
    if not ckpt_path.exists():
        print("  MISSING: Devign ECG RGCN checkpoint", flush=True)
        return None, None, None, None

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    in_dim = state.get("in_dim", 32) if isinstance(state, dict) else 32
    model = RelationalGNN(in_channels=in_dim, hidden=HIDDEN, num_layers=NUM_LAYERS).to(DEVICE)
    model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    model.eval()

    clean_graphs = load_graphs(DEVIGN_IN / "originals_train/test_GGNNinput.json")
    ren_graphs   = load_graphs(DEVIGN_IN / "obf_identifier_test/test_GGNNinput.json")

    def run(graphs):
        loader = PyGLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        probs, labels = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                logits = model(b.x, b.edge_index, b.edge_type, b.batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(b.y.squeeze(-1).long().cpu().numpy().tolist())
        return np.array(probs), np.array(labels)

    pc, lc = run(clean_graphs)
    pr, lr = run(ren_graphs)
    print(f"  Devign ECG RGCN: clean n={len(pc)}, renamed n={len(pr)}", flush=True)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# DEVIGN — REVEAL (Phase 1 logits)
# ──────────────────────────────────────────────

def devign_reveal():
    sys.path.insert(0, str(THESIS / "baselines" / "pyg_gnn"))
    from train_reveal_faithful import load_graphs, RevealGNN, HIDDEN, NUM_BLOCKS
    sys.path.pop(0)

    ckpt_path = THESIS / "baselines/pyg_gnn/models/reveal_faithful/phase1_best.pt"
    if not ckpt_path.exists():
        print("  MISSING: Devign REVEAL checkpoint", flush=True)
        return None, None, None, None

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = RevealGNN(hidden=HIDDEN, num_blocks=NUM_BLOCKS).to(DEVICE)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    clean_graphs = load_graphs(DEVIGN_IN / "originals_train/test_GGNNinput.json")
    ren_graphs   = load_graphs(DEVIGN_IN / "obf_identifier_test/test_GGNNinput.json")

    def run(graphs):
        loader = PyGLoader(graphs, batch_size=BATCH, shuffle=False)
        probs, labels = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                logits = model(b.x, b.edge_index, b.batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(b.y.squeeze(-1).long().cpu().numpy().tolist())
        return np.array(probs), np.array(labels)

    pc, lc = run(clean_graphs)
    pr, lr = run(ren_graphs)
    print(f"  Devign REVEAL: clean n={len(pc)}, renamed n={len(pr)}", flush=True)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# DEVIGN — ReGVD
# ──────────────────────────────────────────────

def devign_regvd():
    regvd_dir = THESIS / "baselines" / "regvd"
    ckpt_path = regvd_dir / "models/regvd_devign/best.pt"
    if not ckpt_path.exists():
        print("  MISSING: Devign ReGVD checkpoint", flush=True)
        return None, None, None, None

    sys.path.insert(0, str(regvd_dir))
    from train_regvd import (
        ReGVDDataset, ReGVDModel, SPLIT_FILE, DATA_FILES,
        CODEBERT_MODEL, HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE, MAX_TOKENS, WINDOW_SIZE
    )
    sys.path.pop(0)

    from transformers import RobertaTokenizer, RobertaModel
    from torch_geometric.loader import DataLoader as PygDL

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_files = set(split["splits"]["test"])

    def load_records(data_path):
        with open(data_path) as f:
            all_recs = json.load(f)
        idx = {d["file_name"]: d for d in all_recs}
        return [idx[fn] for fn in test_files if fn in idx]

    clean_recs = load_records(DATA_FILES["originals"])
    ren_recs   = load_records(DATA_FILES["obf_identifier"])
    print(f"  Devign ReGVD: clean n={len(clean_recs)}, renamed n={len(ren_recs)}", flush=True)

    tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
    del codebert

    model = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    def run(records):
        ds     = ReGVDDataset(records, embed_weight, tokenizer)
        loader = PygDL(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        probs, labels = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(DEVICE)
                logits = model(data)
                p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
                labels.extend(data.y.cpu().numpy().tolist())
        return np.array(probs), np.array(labels)

    pc, lc = run(clean_recs)
    pr, lr = run(ren_recs)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# BIGVUL / DIVERSEVUL — ECG RGCN
# ──────────────────────────────────────────────

def cross_ecg_rgcn(dataset):
    """dataset: 'bigvul' or 'diversevul'"""
    if dataset == 'bigvul':
        cpg_parser = HOME / "bigvul_cpg_parser.py"
        ckpt_path  = HOME / "ecgrgcn_bigvul_ckpts/ecgrgcn_bv_seed42.pt"
        parser_mod = "bigvul_cpg_parser"
    else:
        cpg_parser = HOME / "diversevul_cpg_parser.py"
        ckpt_path  = HOME / "ecgrgcn_diversevul_ckpts/ecgrgcn_dv_seed42.pt"
        parser_mod = "diversevul_cpg_parser"

    if not ckpt_path.exists():
        print(f"  MISSING: {dataset} ECG RGCN checkpoint at {ckpt_path}", flush=True)
        return None, None, None, None

    sys.path.insert(0, str(HOME))
    mod = __import__(parser_mod)
    NUM_NODE_TYPES = mod.NUM_NODE_TYPES
    NUM_EDGE_TYPES = mod.NUM_EDGE_TYPES

    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
    HIDDEN = 128; LAYERS = 3; EMBED = 32; DROP = 0.1

    class ECGRGCNCross(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, EMBED, padding_idx=0)
            self.input_proj = nn.Linear(EMBED, HIDDEN)
            self.convs      = nn.ModuleList([RGCNConv(HIDDEN, HIDDEN, num_relations=NUM_EDGE_TYPES) for _ in range(LAYERS)])
            self.norms      = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
            self.dropout    = nn.Dropout(DROP)
            self.classifier = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(), nn.Dropout(DROP), nn.Linear(HIDDEN, 2))

        def forward(self, data):
            x, ei, et, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if x.dim() > 1: x = x.squeeze(-1)
            x = F.relu(self.input_proj(self.embed(x.long())))
            for conv, norm in zip(self.convs, self.norms):
                x = norm(F.relu(conv(x, ei, et)) + x)
                x = self.dropout(x)
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(g)

    model = ECGRGCNCross().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    print(f"  Loading {dataset} ECG RGCN clean CPG graphs...", flush=True)
    clean_graphs = mod.load_split('test', node_feats='type')
    print(f"  Loading {dataset} ECG RGCN renamed CPG graphs...", flush=True)
    ren_graphs   = mod.load_split('test_obf_identifier', node_feats='type')

    sys.path.pop(0)

    pc, lc = probs_from_pyg(model, PyGLoader(clean_graphs, batch_size=BATCH, shuffle=False))
    pr, lr = probs_from_pyg(model, PyGLoader(ren_graphs,   batch_size=BATCH, shuffle=False))
    print(f"  {dataset} ECG RGCN: clean n={len(pc)}, renamed n={len(pr)}", flush=True)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# BIGVUL / DIVERSEVUL — REVEAL
# ──────────────────────────────────────────────

def cross_reveal(dataset):
    if dataset == 'bigvul':
        parser_mod = "bigvul_cpg_parser"
        ckpt_path  = HOME / "reveal_bigvul_ckpts/reveal_bv_seed42.pt"
    else:
        parser_mod = "reveal_sys_diversevul_fixed_ckpts"
        ckpt_path  = HOME / "reveal_sys_diversevul_fixed_ckpts/reveal_dv_fixed_seed42.pt"
        parser_mod = "diversevul_cpg_parser"

    if not ckpt_path.exists():
        print(f"  MISSING: {dataset} REVEAL checkpoint at {ckpt_path}", flush=True)
        return None, None, None, None

    sys.path.insert(0, str(HOME))
    mod = __import__(parser_mod)
    NUM_NODE_TYPES = mod.NUM_NODE_TYPES

    from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
    HIDDEN = 200; NUM_BLOCKS = 4; STEPS = 2; EMBED = 32; DROP = 0.3

    class REVEALCross(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, EMBED, padding_idx=0)
            self.input_proj = nn.Linear(EMBED, HIDDEN)
            self.blocks     = nn.ModuleList([
                nn.ModuleList([GatedGraphConv(HIDDEN, STEPS), nn.LayerNorm(HIDDEN)])
                for _ in range(NUM_BLOCKS)])
            self.dropout    = nn.Dropout(DROP)
            self.classifier = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(), nn.Dropout(DROP), nn.Linear(HIDDEN, 2))

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x.dim() > 1: x = x.squeeze(-1)
            x = F.relu(self.input_proj(self.embed(x.long())))
            for gru, norm in self.blocks:
                res = x; x = gru(x, edge_index)
                x = norm(x + res); x = F.relu(x); x = self.dropout(x)
            return self.classifier(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1))

    model = REVEALCross().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    print(f"  Loading {dataset} REVEAL clean CPG graphs...", flush=True)
    clean_graphs = mod.load_split('test', node_feats='type')
    print(f"  Loading {dataset} REVEAL renamed CPG graphs...", flush=True)
    ren_graphs   = mod.load_split('test_obf_identifier', node_feats='type')

    sys.path.pop(0)

    pc, lc = probs_from_pyg(model, PyGLoader(clean_graphs, batch_size=BATCH, shuffle=False))
    pr, lr = probs_from_pyg(model, PyGLoader(ren_graphs,   batch_size=BATCH, shuffle=False))
    print(f"  {dataset} REVEAL: clean n={len(pc)}, renamed n={len(pr)}", flush=True)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# BIGVUL / DIVERSEVUL — ReGVD (loads saved probs from train_regvd_roc.py)
# ──────────────────────────────────────────────

def cross_regvd(dataset):
    probs_path = THESIS / "devign_full" / f"{dataset}_regvd_roc_probs.json"
    if not probs_path.exists():
        print(f"  MISSING: {dataset} ReGVD probs at {probs_path} — run train_regvd_roc.py first", flush=True)
        return None, None, None, None
    with open(probs_path) as f:
        data = json.load(f)
    pc = np.array(data["clean"]["probs"])
    lc = np.array(data["clean"]["labels"])
    pr = np.array(data["renamed"]["probs"])
    lr = np.array(data["renamed"]["labels"])
    print(f"  {dataset} ReGVD (saved probs): clean n={len(pc)}, renamed n={len(pr)}", flush=True)
    return pc, lc, pr, lr


# ──────────────────────────────────────────────
# Run all extractions
# ──────────────────────────────────────────────

print("\n=== Devign ===", flush=True)
dv_ecg_pc, dv_ecg_lc, dv_ecg_pr, dv_ecg_lr     = devign_ecg_rgcn()
dv_rev_pc, dv_rev_lc, dv_rev_pr, dv_rev_lr       = devign_reveal()
dv_rgvd_pc, dv_rgvd_lc, dv_rgvd_pr, dv_rgvd_lr  = devign_regvd()

print("\n=== BigVul ===", flush=True)
bv_ecg_pc, bv_ecg_lc, bv_ecg_pr, bv_ecg_lr   = cross_ecg_rgcn('bigvul')
bv_rev_pc, bv_rev_lc, bv_rev_pr, bv_rev_lr    = cross_reveal('bigvul')
bv_rgvd_pc, bv_rgvd_lc, bv_rgvd_pr, bv_rgvd_lr = cross_regvd('bigvul')

print("\n=== DiverseVul ===", flush=True)
div_ecg_pc, div_ecg_lc, div_ecg_pr, div_ecg_lr  = cross_ecg_rgcn('diversevul')
div_rev_pc, div_rev_lc, div_rev_pr, div_rev_lr   = cross_reveal('diversevul')
div_rgvd_pc, div_rgvd_lc, div_rgvd_pr, div_rgvd_lr = cross_regvd('diversevul')


# ──────────────────────────────────────────────
# Plotting helper
# ──────────────────────────────────────────────

COLORS = {"ECG RGCN": "#1f77b4", "REVEAL": "#2ca02c", "ReGVD": "#d62728"}

DATASETS_SPEC = [
    ("Devign",
     dv_ecg_pc,  dv_ecg_lc,  dv_ecg_pr,  dv_ecg_lr,
     dv_rev_pc,  dv_rev_lc,  dv_rev_pr,  dv_rev_lr,
     dv_rgvd_pc, dv_rgvd_lc, dv_rgvd_pr, dv_rgvd_lr),
    ("Big-Vul",
     bv_ecg_pc,  bv_ecg_lc,  bv_ecg_pr,  bv_ecg_lr,
     bv_rev_pc,  bv_rev_lc,  bv_rev_pr,  bv_rev_lr,
     bv_rgvd_pc, bv_rgvd_lc, bv_rgvd_pr, bv_rgvd_lr),
    ("DiverseVul",
     div_ecg_pc, div_ecg_lc, div_ecg_pr, div_ecg_lr,
     div_rev_pc, div_rev_lc, div_rev_pr, div_rev_lr,
     div_rgvd_pc, div_rgvd_lc, div_rgvd_pr, div_rgvd_lr),
]


def draw_dataset_ax(ax, title,
                    ecg_pc, ecg_lc, ecg_pr, ecg_lr,
                    rev_pc, rev_lc, rev_pr, rev_lr,
                    rgvd_pc, rgvd_lc, rgvd_pr, rgvd_lr):
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, label='Random (AUC=0.50)')
    for name, pc, lc, pr, lr in [("ECG RGCN", ecg_pc, ecg_lc, ecg_pr, ecg_lr),
                                   ("REVEAL",   rev_pc, rev_lc, rev_pr, rev_lr),
                                   ("ReGVD",    rgvd_pc, rgvd_lc, rgvd_pr, rgvd_lr)]:
        color = COLORS[name]
        if pc is not None:
            r = roc_from_probs(pc, lc, name, "clean")
            if r:
                ax.plot(r[0], r[1], color=color, lw=2, linestyle='-',
                        label=f"{name} Clean (AUC={r[2]:.3f})")
        if pr is not None:
            r = roc_from_probs(pr, lr, name, "renamed")
            if r:
                ax.plot(r[0], r[1], color=color, lw=2, linestyle='--',
                        label=f"{name} Rename (AUC={r[2]:.3f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)


# ── Combined 3-panel figure ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, spec in zip(axes, DATASETS_SPEC):
    draw_dataset_ax(ax, *spec)
fig.suptitle("ROC Under Identifier Renaming Across Datasets", fontsize=15, fontweight='bold')
plt.tight_layout()
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"roc_renaming_across_datasets.{ext}"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    print(f"Saved: {out}", flush=True)
plt.close()


# ── Three separate per-dataset figures ──────
DS_KEYS = ["devign", "bigvul", "diversevul"]
for key, spec in zip(DS_KEYS, DATASETS_SPEC):
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    draw_dataset_ax(ax2, *spec)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"roc_renaming_{key}.{ext}"
        plt.savefig(out, bbox_inches='tight', dpi=150)
        print(f"Saved: {out}", flush=True)
    plt.close()

print("Done.", flush=True)
