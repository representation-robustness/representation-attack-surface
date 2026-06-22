#!/usr/bin/env python3
"""
Extract per-function predictions for ECG RGCN, VulGNN, and ANGLE on all 7
obfuscation conditions. Saves one JSON per model to attack/preds/.

Output format:
{
  "model": "ecg_rgcn",
  "conditions": ["clean", "ren", "dead", "cf", "ren_dead", "ren_cf", "dead_cf", "compound"],
  "true_labels": [1, 0, ...],       # N_test integers
  "seeds": {
    "42": {
      "clean": [1, 0, ...], "ren": [...], ...   # predicted labels per condition
    }, ...
  }
}
"""

import json, os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGPooling, GCNConv, TransformerConv
from sklearn.metrics import f1_score

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 256
CPG_ROOT = os.path.expanduser("~/vul-LMGGNN/data/cpg")
OUT_DIR  = Path(__file__).parent / "preds"
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "clean":    "test",
    "ren":      "test_obf_identifier",
    "dead":     "test_obf_deadcode",
    "cf":       "test_obf_controlflow",
    "ren_dead": "test_obf_ren_dead",
    "ren_cf":   "test_obf_ren_cf",
    "dead_cf":  "test_obf_dead_cf",
    "compound": "test_obf_compound",
}

# ──────────────────────────────────────────────
# ECG RGCN
# ──────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/vulgnn_devign"))
from cpg_parser import cpg_to_graph, NUM_NODE_TYPES, NUM_EDGE_TYPES

ECG_HIDDEN = 128; ECG_LAYERS = 3; ECG_EMBED = 32; ECG_DROP = 0.1

class ECGRGCNDevign(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, ECG_EMBED, padding_idx=0)
        self.input_proj = nn.Linear(ECG_EMBED, ECG_HIDDEN)
        self.convs = nn.ModuleList([RGCNConv(ECG_HIDDEN, ECG_HIDDEN, num_relations=NUM_EDGE_TYPES)
                                    for _ in range(ECG_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(ECG_HIDDEN) for _ in range(ECG_LAYERS)])
        self.dropout    = nn.Dropout(ECG_DROP)
        self.classifier = nn.Sequential(
            nn.Linear(ECG_HIDDEN * 2, ECG_HIDDEN), nn.ReLU(),
            nn.Dropout(ECG_DROP), nn.Linear(ECG_HIDDEN, 2))

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dim() > 1: x = x.squeeze(-1)
        x = F.relu(self.input_proj(self.embed(x.long())))
        for conv, norm in zip(self.convs, self.norms):
            res = x; x = conv(x, edge_index, edge_type)
            x = norm(x + res); x = F.relu(x); x = self.dropout(x)
        return self.classifier(torch.cat([global_mean_pool(x, batch),
                                           global_max_pool(x, batch)], dim=-1))


def load_cpg_split(split_name):
    split_dir = os.path.join(CPG_ROOT, split_name)
    pkls = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    graphs = []
    for p in pkls:
        df = pd.read_pickle(p)
        for _, row in df.iterrows():
            g = cpg_to_graph(row["cpg"], row["func"], int(row["target"]))
            if g is not None:
                graphs.append(g)
    return graphs


@torch.no_grad()
def predict_labels(model, loader):
    model.eval()
    preds, truths = [], []
    for b in loader:
        b = b.to(DEVICE)
        preds.extend(model(b).argmax(-1).cpu().tolist())
        truths.extend(b.y.long().cpu().tolist())
    return preds, truths


def extract_ecg_rgcn():
    print("\n=== ECG RGCN ===", flush=True)
    ckpt_dir = os.path.expanduser("~/ecgrgcn_devign_ckpts")
    seeds = [42, 1337, 7, 100, 999]

    print("Loading test splits...", flush=True)
    cond_graphs = {}
    for cond, split in CONDITIONS.items():
        cond_graphs[cond] = load_cpg_split(split)
        print(f"  {cond}: {len(cond_graphs[cond])}", flush=True)

    true_labels = [int(g.y.item()) for g in cond_graphs["clean"]]
    result = {"model": "ecg_rgcn", "conditions": list(CONDITIONS.keys()),
              "true_labels": true_labels, "seeds": {}}

    for seed in seeds:
        ckpt = os.path.join(ckpt_dir, f"ecgrgcn_dv_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  Seed {seed}: missing checkpoint, skip", flush=True); continue
        model = ECGRGCNDevign().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        print(f"  Seed {seed}", flush=True)
        seed_preds = {}
        for cond, graphs in cond_graphs.items():
            loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            preds, _ = predict_labels(model, loader)
            seed_preds[cond] = preds
            n = min(len(true_labels), len(preds))
            print(f"    {cond}: F1={f1_score(true_labels[:n], preds[:n], zero_division=0)*100:.2f}% ({len(preds)} samples)", flush=True)
        result["seeds"][str(seed)] = seed_preds
        del model; torch.cuda.empty_cache()

    out = OUT_DIR / "ecg_rgcn_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


# ──────────────────────────────────────────────
# VulGNN
# ──────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/vulgnn_devign"))
from model import VulGNN


def extract_vulgnn():
    print("\n=== VulGNN ===", flush=True)
    ckpt_dir = os.path.expanduser("~/vulgnn_devign/checkpoints")
    seeds = [42, 1337, 7, 100, 999]

    print("Loading test splits...", flush=True)
    cond_graphs = {}
    for cond, split in CONDITIONS.items():
        cond_graphs[cond] = load_cpg_split(split)
        print(f"  {cond}: {len(cond_graphs[cond])}", flush=True)

    true_labels = [int(g.y.item()) for g in cond_graphs["clean"]]
    result = {"model": "vulgnn", "conditions": list(CONDITIONS.keys()),
              "true_labels": true_labels, "seeds": {}}

    # infer model dims from first graph
    sample = cond_graphs["clean"][0]
    in_dim = sample.x.shape[-1] if sample.x.dim() > 1 else 1

    for seed in seeds:
        ckpt = os.path.join(ckpt_dir, f"vulgnn_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  Seed {seed}: missing, skip", flush=True); continue

        # load model - peek at checkpoint to get architecture
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = VulGNN(num_node_types=NUM_NODE_TYPES, num_edge_types=NUM_EDGE_TYPES).to(DEVICE)
        model.load_state_dict(state)
        print(f"  Seed {seed}", flush=True)

        seed_preds = {}
        for cond, graphs in cond_graphs.items():
            loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            preds, _ = predict_labels(model, loader)
            seed_preds[cond] = preds
            n = min(len(true_labels), len(preds))
            print(f"    {cond}: F1={f1_score(true_labels[:n], preds[:n], zero_division=0)*100:.2f}% ({len(preds)} samples)", flush=True)
        result["seeds"][str(seed)] = seed_preds
        del model; torch.cuda.empty_cache()

    out = OUT_DIR / "vulgnn_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


# ──────────────────────────────────────────────
# ANGLE
# ──────────────────────────────────────────────
def extract_angle():
    print("\n=== ANGLE ===", flush=True)

    # ANGLE needs its own cpg_parser (uses W2V tokenization)
    angle_dir = os.path.expanduser("~/angle_devign")
    sys.path.insert(0, angle_dir)
    import importlib
    if "cpg_parser" in sys.modules:
        del sys.modules["cpg_parser"]
    from cpg_parser import load_split as angle_load_split
    from model import ANGLE as ANGLEModel
    import json as _json
    from gensim.models import KeyedVectors

    ckpt_dir = os.path.join(angle_dir, "checkpoints")
    seeds = [42, 1337, 7, 100, 999]
    HIDDEN = 64; NUM_LAYERS = 3; POOL_RATIO = 0.5; DROPOUT = 0.1

    vocab = _json.load(open(os.path.join(angle_dir, "vocab.json")))
    wv    = KeyedVectors.load(os.path.join(angle_dir, "w2v_model.bin"), mmap='r')
    embed_dim  = wv.vector_size
    vocab_size = len(vocab)
    pretrained = torch.zeros(vocab_size + 1, embed_dim)
    for word, idx in vocab.items():
        if word in wv:
            pretrained[idx] = torch.tensor(wv[word])

    print("Loading test splits...", flush=True)
    cond_graphs = {}
    for cond, split in CONDITIONS.items():
        try:
            cond_graphs[cond] = angle_load_split(split)
            print(f"  {cond}: {len(cond_graphs[cond])}", flush=True)
        except Exception as e:
            print(f"  SKIP {cond}: {e}", flush=True)

    true_labels = [int(g.y.item()) for g in cond_graphs["clean"]]
    result = {"model": "angle", "conditions": list(cond_graphs.keys()),
              "true_labels": true_labels, "seeds": {}}

    for seed in seeds:
        ckpt = os.path.join(ckpt_dir, f"angle_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  Seed {seed}: missing, skip", flush=True); continue
        model = ANGLEModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden=HIDDEN,
                           pool_ratio=POOL_RATIO, num_layers=NUM_LAYERS, dropout=DROPOUT,
                           pretrained_emb=pretrained).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        print(f"  Seed {seed}", flush=True)

        seed_preds = {}
        for cond, graphs in cond_graphs.items():
            loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            preds, _ = predict_labels(model, loader)
            seed_preds[cond] = preds
            n = min(len(true_labels), len(preds))
            print(f"    {cond}: F1={f1_score(true_labels[:n], preds[:n], zero_division=0)*100:.2f}% ({len(preds)} samples)", flush=True)
        result["seeds"][str(seed)] = seed_preds
        del model; torch.cuda.empty_cache()

    out = OUT_DIR / "angle_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ecg_rgcn", "vulgnn", "angle", "all"])
    args = ap.parse_args()
    print(f"Device: {DEVICE}", flush=True)
    if args.model in ("ecg_rgcn", "all"): extract_ecg_rgcn()
    if args.model in ("vulgnn", "all"):   extract_vulgnn()
    if args.model in ("angle", "all"):    extract_angle()
