"""
eval_7cond_devign_gnns.py — Evaluate GNN models on all 8 conditions for Devign.

Models: ECG-RGCN (REVEAL has no Devign checkpoints)
Checkpoints: ~/ecgrgcn_devign_ckpts/

Usage:
    CUDA_VISIBLE_DEVICES=X python eval_7cond_devign_gnns.py --model ecgrgcn

Output: ~/thesis/devign_full/{model}_7cond_results.json
"""

import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

THESIS_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(THESIS_ROOT / "baselines" / "vulgnn"))
from cpg_parser import cpg_to_graph, NUM_NODE_TYPES, NUM_EDGE_TYPES

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH       = 128
SEEDS       = [42, 1337, 7, 100, 999]
CPG_ROOT    = os.path.expanduser("~/vul-LMGGNN/data/cpg")
RESULTS_DIR = str(THESIS_ROOT / "devign_full")

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
    'ecgrgcn': os.path.expanduser('~/ecgrgcn_devign_ckpts'),
}

CKPT_PATTERNS = {
    'ecgrgcn': 'ecgrgcn_dv_seed{seed}.pt',
}


class ECGRGCNDevign(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, 32, padding_idx=0)
        self.input_proj = nn.Linear(32, 128)
        self.convs      = nn.ModuleList([
            RGCNConv(128, 128, num_relations=NUM_EDGE_TYPES) for _ in range(3)
        ])
        self.norms      = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2),
        )

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dim() > 1: x = x.squeeze(-1)
        x = F.relu(self.input_proj(self.embed(x.long())))
        for conv, norm in zip(self.convs, self.norms):
            res = x; x = conv(x, edge_index, edge_type)
            x = norm(x + res); x = F.relu(x); x = self.dropout(x)
        return self.classifier(torch.cat([global_mean_pool(x, batch),
                                          global_max_pool(x, batch)], dim=-1))


def load_split(split_name):
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
    print(f"Model: {model_name.upper()}  |  Dataset: Devign  |  Device: {DEVICE}")

    print("\nLoading test splits...", flush=True)
    test_loaders = {}
    for cond_name, split_name in ALL_SPLITS.items():
        try:
            graphs = load_split(split_name)
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
            print(f"\nSeed {seed}: checkpoint not found at {ckpt_path}, skipping", flush=True)
            continue

        print(f"\n{'-'*50}\nSeed {seed}\n{'-'*50}", flush=True)
        model = ECGRGCNDevign().to(DEVICE)
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

        out = os.path.join(RESULTS_DIR, f'{model_name}_7cond_devign_seed{seed}_results.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        all_results.append(results)

    if not all_results:
        print("No results.", flush=True)
        return

    conditions = list(test_loaders.keys())
    agg = aggregate(all_results, conditions, model_name.upper())

    out = os.path.join(RESULTS_DIR, f'{model_name}_7cond_results.json')
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
    parser.add_argument('--model', required=True, choices=['ecgrgcn'])
    args = parser.parse_args()
    run_model(args.model)
