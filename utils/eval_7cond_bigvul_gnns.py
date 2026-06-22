"""
eval_7cond_bigvul_gnns.py — Evaluate GNN models on all 8 conditions for Big-Vul.

Models: VulGNN, ANGLE, ECG-RGCN, REVEAL
Checkpoints: ~/vulgnn_bigvul_ckpts/, ~/angle_bigvul_ckpts/,
             ~/ecgrgcn_bigvul_ckpts/, ~/reveal_bigvul_ckpts/

Usage:
    CUDA_VISIBLE_DEVICES=X python eval_7cond_bigvul_gnns.py --model vulgnn
    CUDA_VISIBLE_DEVICES=X python eval_7cond_bigvul_gnns.py --model angle
    CUDA_VISIBLE_DEVICES=X python eval_7cond_bigvul_gnns.py --model ecgrgcn
    CUDA_VISIBLE_DEVICES=X python eval_7cond_bigvul_gnns.py --model reveal

Output: ~/thesis/devign_full/{model}_7cond_bigvul_results.json
"""

import argparse, json, os, sys
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, GatedGraphConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

THESIS_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(THESIS_ROOT / "utils"))
from bigvul_cpg_parser import load_split, NUM_NODE_TYPES, NUM_EDGE_TYPES

sys.path.insert(0, str(THESIS_ROOT / "baselines" / "vulgnn"))
from model import VulGNN

def _load_angle_class():
    spec = importlib.util.spec_from_file_location(
        "angle_model", str(THESIS_ROOT / "baselines" / "angle" / "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ANGLE

ANGLE = _load_angle_class()

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH       = 128
SEEDS       = [42, 1337, 7, 100, 999]
RESULTS_DIR = str(THESIS_ROOT / "devign_full")
W2V_PATH    = str(THESIS_ROOT / "baselines" / "angle" / "w2v_model.bin")
VOCAB_PATH  = str(THESIS_ROOT / "baselines" / "angle" / "vocab.json")

ALL_SPLITS = {
    'test':                 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
    'test_obf_ren_dead':    'test_obf_ren_dead',
    'test_obf_ren_cf':      'test_obf_ren_cf',
    'test_obf_dead_cf':     'test_obf_dead_cf',
    'test_obf_compound':    'test_obf_compound',
}

CKPT_DIRS = {
    'vulgnn':  os.path.expanduser('~/vulgnn_bigvul_ckpts'),
    'angle':   os.path.expanduser('~/angle_bigvul_ckpts'),
    'ecgrgcn': os.path.expanduser('~/ecgrgcn_bigvul_ckpts'),
    'reveal':  os.path.expanduser('~/reveal_bigvul_ckpts'),
}

CKPT_PATTERNS = {
    'vulgnn':  'vulgnn_bv_seed{seed}.pt',
    'angle':   'angle_bv_seed{seed}.pt',
    'ecgrgcn': 'ecgrgcn_bv_seed{seed}.pt',
    'reveal':  'reveal_bv_seed{seed}.pt',
}


# ── Inline model classes (matching training scripts exactly) ──────────────────

class ECGRGCNBigVul(nn.Module):
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
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dim() > 1: x = x.squeeze(-1)
        x = F.relu(self.input_proj(self.embed(x.long())))
        for conv, norm in zip(self.convs, self.norms):
            res = x; x = conv(x, edge_index, edge_type); x = norm(x + res); x = F.relu(x); x = self.dropout(x)
        return self.classifier(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1))


class REVEALBigVul(nn.Module):
    def __init__(self, num_node_types, embed_dim, hidden, num_blocks, steps_per, dropout):
        super().__init__()
        self.embed      = nn.Embedding(num_node_types + 1, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, hidden)
        self.blocks     = nn.ModuleList([
            nn.ModuleList([GatedGraphConv(hidden, steps_per), nn.LayerNorm(hidden)])
            for _ in range(num_blocks)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.dim() > 1: x = x.squeeze(-1)
        x = F.relu(self.input_proj(self.embed(x.long())))
        for gru, norm in self.blocks:
            res = x; x = gru(x, edge_index); x = norm(x + res); x = F.relu(x); x = self.dropout(x)
        return self.classifier(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1))


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            y_true.extend(batch.y.squeeze().long().cpu().tolist())
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rc  = recall_score(y_true, y_pred, zero_division=0)
    return f1, acc, pr, rc


def load_w2v():
    from gensim.models import KeyedVectors
    vocab = json.load(open(VOCAB_PATH))
    wv    = KeyedVectors.load(W2V_PATH, mmap='r')
    embed_dim  = wv.vector_size
    vocab_size = len(vocab)
    pretrained = torch.zeros(vocab_size + 1, embed_dim)
    for word, idx in vocab.items():
        if word in wv:
            pretrained[idx] = torch.tensor(wv[word])
    return vocab, embed_dim, vocab_size, pretrained


def build_model(model_name, vocab_size=None, embed_dim=None, pretrained=None):
    if model_name == 'vulgnn':
        return VulGNN(num_node_types=NUM_NODE_TYPES, embed_dim=16,
                      hidden=128, num_layers=6, dropout=0.08,
                      num_edge_types=NUM_EDGE_TYPES, edge_dim=4).to(DEVICE)
    elif model_name == 'angle':
        return ANGLE(vocab_size=vocab_size, embed_dim=embed_dim, hidden=64,
                     pool_ratio=0.5, num_layers=3, dropout=0.1,
                     pretrained_emb=pretrained).to(DEVICE)
    elif model_name == 'ecgrgcn':
        return ECGRGCNBigVul(NUM_NODE_TYPES, embed_dim=32, hidden=128,
                             num_layers=3, dropout=0.1).to(DEVICE)
    elif model_name == 'reveal':
        return REVEALBigVul(NUM_NODE_TYPES, embed_dim=32, hidden=200,
                            num_blocks=4, steps_per=2, dropout=0.3).to(DEVICE)


def aggregate(all_results, conditions, model_label):
    agg = {}
    for cond in conditions:
        f1s = [r[cond]['f1'] for r in all_results if cond in r]
        if not f1s:
            continue
        agg[cond] = {
            'f1_mean': round(float(np.mean(f1s)), 2),
            'f1_std':  round(float(np.std(f1s)), 2),
            'all_f1':  f1s,
        }
    base_f1 = agg.get('test', {}).get('f1_mean', 0)
    for cond in conditions:
        if cond != 'test' and cond in agg:
            agg[cond]['delta_f1'] = round(agg[cond]['f1_mean'] - base_f1, 2)
    agg['n_seeds'] = len(all_results)
    agg['seeds']   = [r['seed'] for r in all_results]
    agg['model']   = model_label
    return agg


def run_model(model_name):
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}  |  Dataset: Big-Vul  |  Device: {DEVICE}")

    # ANGLE needs W2V; others use node type integers
    node_feats = 'code' if model_name == 'angle' else 'type'
    vocab_size = embed_dim_w2v = pretrained = None
    if model_name == 'angle':
        print("Loading W2V...", flush=True)
        _, embed_dim_w2v, vocab_size, pretrained = load_w2v()

    print("\nLoading test splits...", flush=True)
    test_loaders = {}
    for cond_name, split_name in ALL_SPLITS.items():
        try:
            graphs = load_split(split_name, node_feats=node_feats)
            test_loaders[cond_name] = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            print(f"  {cond_name}: {len(graphs)}", flush=True)
        except Exception as e:
            print(f"  SKIP {cond_name}: {e}", flush=True)

    ckpt_dir     = CKPT_DIRS[model_name]
    ckpt_pattern = CKPT_PATTERNS[model_name]
    all_results  = []

    for seed in SEEDS:
        ckpt_path = os.path.join(ckpt_dir, ckpt_pattern.format(seed=seed))
        if not os.path.exists(ckpt_path):
            print(f"\nSeed {seed}: checkpoint not found, skipping", flush=True)
            continue

        print(f"\n{'-'*50}\nSeed {seed}\n{'-'*50}", flush=True)
        model = build_model(model_name, vocab_size, embed_dim_w2v, pretrained)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        model.eval()

        results = {'seed': seed}
        for cond_name, loader in test_loaders.items():
            f1, acc, pr, rc = evaluate(model, loader)
            results[cond_name] = {
                'f1':  round(f1 * 100, 2),
                'acc': round(acc * 100, 2),
                'pr':  round(pr * 100, 2),
                'rc':  round(rc * 100, 2),
            }
            print(f"  {cond_name}: F1={f1*100:.2f}%", flush=True)

        out = os.path.join(RESULTS_DIR, f'{model_name}_7cond_bigvul_seed{seed}_results.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        all_results.append(results)

    if not all_results:
        print("No results.", flush=True)
        return

    conditions = list(test_loaders.keys())
    agg = aggregate(all_results, conditions, model_name.upper())

    out = os.path.join(RESULTS_DIR, f'{model_name}_7cond_bigvul_results.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated → {out}", flush=True)

    print(f"\n{'Condition':<28} {'F1':>8} {'StD':>8} {'ΔF1':>8}")
    for cond in conditions:
        if cond not in agg:
            continue
        m = agg[cond]
        delta = f"{m.get('delta_f1', 0):+.2f}" if cond != 'test' else "—"
        print(f"{cond:<28} {m['f1_mean']:>7.2f}% {m['f1_std']:>7.2f}% {delta:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vulgnn', 'angle', 'ecgrgcn', 'reveal'])
    args = parser.parse_args()
    run_model(args.model)
