"""
lmggnn_bigvul.py — Vul-LMGGNN training on Big-Vul.

Adapted from ~/vul-LMGGNN/run_multiseed.py.
Runs 5 seeds and aggregates results.

Usage:
    CUDA_VISIBLE_DEVICES=4 python ~/lmggnn_bigvul.py

Outputs:
    ~/thesis/devign_full/bigvul_lmggnn_multiseed_results.json
"""

import json, os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

LMGGNN_DIR = os.path.expanduser("~/vul-LMGGNN")
sys.path.insert(0, LMGGNN_DIR)
sys.path.insert(0, os.path.join(LMGGNN_DIR, "models"))
sys.path.insert(0, os.path.expanduser("~"))

from bigvul_cpg_parser import load_split

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 20
BATCH      = 8
RESULTS_DIR = os.path.expanduser('~/thesis/devign_full')
CKPT_DIR    = os.path.expanduser('~/lmggnn_bigvul_ckpts')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 1337, 7, 100, 999]

SPLITS = {
    'test':                 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
}


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
            truths.extend(b.y.cpu().tolist())
    return (f1_score(truths, preds, zero_division=0),
            accuracy_score(truths, preds),
            precision_score(truths, preds, zero_division=0),
            recall_score(truths, preds, zero_division=0))


def train_one_seed(seed, train_graphs, valid_graphs, test_graphs, Model, model_kwargs, criterion_fn):
    set_seed(seed)
    print(f"\n{'='*55}\n  Vul-LMGGNN Big-Vul  Seed {seed}\n{'='*55}")
    ckpt = os.path.join(CKPT_DIR, f'lmggnn_bv_seed{seed}.pt')

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH)
    test_loaders = {k: DataLoader(v, batch_size=BATCH) for k, v in test_graphs.items()}

    model = Model(**model_kwargs).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0
        for b in train_loader:
            b = b.to(DEVICE); optimizer.zero_grad()
            logits = model(b); loss = criterion_fn(logits, b.y.long().squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()
        val_f1, _, _, _ = evaluate(model, valid_loader)
        print(f"  Ep {epoch:3d}/{EPOCHS} loss={total_loss/max(len(train_loader),1):.4f} val_F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    results = {'seed': seed, 'best_val_f1': round(best_val_f1*100, 2)}
    for split_name, loader in test_loaders.items():
        f1, acc, pr, rc = evaluate(model, loader)
        results[split_name] = {'f1': round(f1*100,2), 'acc': round(acc*100,2),
                                'pr': round(pr*100,2), 'rc': round(rc*100,2)}
        print(f"  {split_name}: F1={f1*100:.2f}%")

    out = os.path.join(RESULTS_DIR, f'bigvul_lmggnn_seed{seed}_results.json')
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
                'model': 'Vul-LMGGNN', 'dataset': 'Big-Vul'})
    return agg


def main():
    print(f"Device: {DEVICE}")

    # Import LMGGNN model
    try:
        from run_multiseed import LMGNNModel
        Model = LMGNNModel
        model_kwargs = {}
    except ImportError:
        # Fallback: use the train_devign model class
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lmggnn_model", os.path.join(LMGGNN_DIR, "models", "LMGNN.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Model = getattr(mod, 'BertGGCN', getattr(mod, 'VulLMGNN', getattr(mod, 'LMGGNN', None)))
        model_kwargs = {}

    if Model is None:
        raise ImportError("Could not import Vul-LMGGNN model class")

    print("Loading Big-Vul CPG splits...")
    t0 = time.time()
    train_graphs = load_split('train')
    valid_graphs = load_split('valid')
    test_graphs  = {k: load_split(v) for k, v in SPLITS.items()}
    print(f"Loaded in {time.time()-t0:.1f}s: train={len(train_graphs)}, valid={len(valid_graphs)}")

    labels = [int(g.y.item()) for g in train_graphs]
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    total = n_pos + n_neg
    w = torch.tensor([total/(2*n_neg), total/(2*n_pos)], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, train_graphs, valid_graphs, test_graphs,
                           Model, model_kwargs, criterion)
        all_results.append(r)

    agg = aggregate(all_results)
    out = os.path.join(RESULTS_DIR, 'bigvul_lmggnn_multiseed_results.json')
    with open(out, 'w') as f: json.dump(agg, f, indent=2)
    print(f"\nResults → {out}")


if __name__ == '__main__':
    main()
