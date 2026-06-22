"""
eval_7cond.py — ANGLE eval-only on all 7 obfuscation conditions.

Loads each saved checkpoint (one per seed), evaluates on all 7 test splits.

Usage:
    CUDA_VISIBLE_DEVICES=X python eval_7cond.py
"""

import json, os, sys, random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from gensim.models import KeyedVectors

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, tokenize_code
from model import ANGLE

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH       = 128
HIDDEN      = 64
NUM_LAYERS  = 3
POOL_RATIO  = 0.5
DROPOUT     = 0.1
RESULTS_DIR = os.path.expanduser('~/thesis/devign_full')
_DIR        = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = os.path.join(_DIR, 'checkpoints')
W2V_PATH    = os.path.join(_DIR, 'w2v_model.bin')
VOCAB_PATH  = os.path.join(_DIR, 'vocab.json')
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


def load_w2v():
    vocab = json.load(open(VOCAB_PATH))
    wv    = KeyedVectors.load(W2V_PATH, mmap='r')
    embed_dim  = wv.vector_size
    vocab_size = len(vocab)
    pretrained = torch.zeros(vocab_size + 1, embed_dim)
    for word, idx in vocab.items():
        if word in wv:
            pretrained[idx] = torch.tensor(wv[word])
    return vocab, embed_dim, vocab_size, pretrained


def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(batch.y.cpu().tolist())
    f1  = f1_score(truths, preds, zero_division=0)
    acc = accuracy_score(truths, preds)
    pr  = precision_score(truths, preds, zero_division=0)
    rc  = recall_score(truths, preds, zero_division=0)
    return f1, acc, pr, rc


def main():
    print(f"Device: {DEVICE}", flush=True)

    print("Loading W2V...", flush=True)
    vocab, embed_dim, vocab_size, pretrained = load_w2v()

    print("\nLoading test splits...", flush=True)
    test_loaders = {}
    for cond_name, split_name in ALL_SPLITS.items():
        try:
            graphs = load_split(split_name)
            test_loaders[cond_name] = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            print(f"  {cond_name}: {len(graphs)}", flush=True)
        except Exception as e:
            print(f"  SKIP {cond_name}: {e}", flush=True)

    all_results = []
    for seed in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f'angle_seed{seed}.pt')
        if not os.path.exists(ckpt_path):
            print(f"\nSeed {seed}: checkpoint not found, skipping", flush=True)
            continue

        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}", flush=True)
        model = ANGLE(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden=HIDDEN,
            pool_ratio=POOL_RATIO,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            pretrained_emb=pretrained,
        ).to(DEVICE)
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

        out = os.path.join(RESULTS_DIR, f'angle_7cond_seed{seed}_results.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        all_results.append(results)

    if not all_results:
        print("No results.", flush=True)
        return

    conditions = list(test_loaders.keys())
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
    agg['model']   = 'ANGLE'

    out = os.path.join(RESULTS_DIR, 'angle_7cond_results.json')
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
    main()
