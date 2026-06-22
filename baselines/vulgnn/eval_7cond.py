"""
eval_7cond.py — VulGNN eval-only on all 7 obfuscation conditions.

Loads each saved checkpoint (one per seed), evaluates on all 7 test splits,
and writes extended multiseed results.

Usage:
    CUDA_VISIBLE_DEVICES=X python eval_7cond.py
"""

import json, os, sys, random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, NUM_NODE_TYPES, NUM_EDGE_TYPES
from model import VulGNN

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH       = 128
RESULTS_DIR = os.path.expanduser('~/thesis/devign_full')
CKPT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
SEEDS       = [42, 1337, 7, 100, 999]

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


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.squeeze().long().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rc  = recall_score(y_true, y_pred, zero_division=0)
    return f1, acc, pr, rc


def build_model():
    return VulGNN(
        num_node_types=NUM_NODE_TYPES,
        embed_dim=16,
        hidden=128,
        num_layers=6,
        dropout=0.08,
        num_edge_types=NUM_EDGE_TYPES,
        edge_dim=4,
    ).to(DEVICE)


def main():
    print(f"Device: {DEVICE}", flush=True)

    print("\nLoading test splits...", flush=True)
    test_graphs = {}
    for cond_name, split_name in ALL_SPLITS.items():
        try:
            graphs = load_split(split_name)
            test_graphs[cond_name] = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            print(f"  {cond_name}: {len(graphs)}", flush=True)
        except Exception as e:
            print(f"  SKIP {cond_name}: {e}", flush=True)

    all_results = []
    for seed in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f'vulgnn_seed{seed}.pt')
        if not os.path.exists(ckpt_path):
            print(f"\nSeed {seed}: checkpoint not found, skipping", flush=True)
            continue

        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}", flush=True)
        model = build_model()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        model.eval()

        results = {'seed': seed}
        for cond_name, loader in test_graphs.items():
            f1, acc, pr, rc = evaluate(model, loader)
            results[cond_name] = {
                'f1':  round(f1 * 100, 2),
                'acc': round(acc * 100, 2),
                'pr':  round(pr * 100, 2),
                'rc':  round(rc * 100, 2),
            }
            print(f"  {cond_name}: F1={f1*100:.2f}%", flush=True)

        out = os.path.join(RESULTS_DIR, f'vulgnn_7cond_seed{seed}_results.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out}", flush=True)
        all_results.append(results)

    if not all_results:
        print("No results collected.", flush=True)
        return

    # Aggregate
    conditions = list(test_graphs.keys())
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
    agg['model']   = 'VulGNN'

    out = os.path.join(RESULTS_DIR, 'vulgnn_7cond_results.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results → {out}", flush=True)

    print(f"\n{'Condition':<28} {'F1':>8} {'StD':>8} {'ΔF1':>8}")
    for cond in conditions:
        if cond not in agg:
            continue
        m = agg[cond]
        delta = f"{m.get('delta_f1', 0):+.2f}" if cond != 'test' else "—"
        print(f"{cond:<28} {m['f1_mean']:>7.2f}% {m['f1_std']:>7.2f}% {delta:>8}")


if __name__ == "__main__":
    main()
