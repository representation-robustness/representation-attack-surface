"""
run_multiseed.py — Run ANGLE training with 5 random seeds and aggregate results.

Usage:
    CUDA_VISIBLE_DEVICES=3 python run_multiseed.py

Outputs:
    ~/thesis/devign_full/angle_seed_{seed}_results.json  (per seed)
    ~/thesis/devign_full/angle_multiseed_results.json    (aggregated)
"""

import json
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from gensim.models import KeyedVectors

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, tokenize_code
from model import ANGLE

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY= 1e-5
BATCH       = 64
HIDDEN      = 64
NUM_LAYERS  = 3
POOL_RATIO  = 0.5
DROPOUT     = 0.1
MAX_SEQ_LEN = 16

_DIR        = os.path.dirname(os.path.abspath(__file__))
W2V_PATH    = os.path.join(_DIR, 'w2v_model.bin')
VOCAB_PATH  = os.path.join(_DIR, 'vocab.json')
RESULTS_DIR = os.path.expanduser('~/thesis/devign_full')
CKPT_DIR    = os.path.join(_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 1337, 7, 100, 999]

SPLITS = {
    'test':                 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_w2v():
    if not (os.path.exists(W2V_PATH) and os.path.exists(VOCAB_PATH)):
        print("W2V model not found — training Word2Vec first...", flush=True)
        import train_w2v
        train_w2v.main()


def load_w2v():
    ensure_w2v()
    vocab = json.load(open(VOCAB_PATH))
    wv    = KeyedVectors.load(W2V_PATH, mmap='r')
    embed_dim  = wv.vector_size
    vocab_size = len(vocab)
    pretrained = torch.zeros(vocab_size + 1, embed_dim)
    for word, idx in vocab.items():
        if word in wv:
            pretrained[idx] = torch.tensor(wv[word])
    return vocab, embed_dim, vocab_size, pretrained


def compute_class_weights(loader):
    labels = []
    for batch in loader:
        labels.extend(batch.y.tolist())
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    total = n_pos + n_neg
    w = torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float)
    return w.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            pred = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(pred)
            truths.extend(batch.y.cpu().tolist())
    f1  = f1_score(truths, preds, zero_division=0)
    acc = accuracy_score(truths, preds)
    pr  = precision_score(truths, preds, zero_division=0)
    rc  = recall_score(truths, preds, zero_division=0)
    return f1, acc, pr, rc


def train_one_seed(seed, train_graphs, valid_graphs, test_graphs, vocab, embed_dim, vocab_size, pretrained):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  ANGLE Seed {seed}")
    print(f"{'='*60}")

    ckpt_path = os.path.join(CKPT_DIR, f'angle_seed{seed}.pt')

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH)
    test_loaders = {k: DataLoader(v, batch_size=BATCH) for k, v in test_graphs.items()}

    model = ANGLE(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden=HIDDEN,
        pool_ratio=POOL_RATIO,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_emb=pretrained,
    ).to(DEVICE)

    weights   = compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, batch.y.squeeze())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        val_f1, val_acc, _, _ = evaluate(model, valid_loader)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Ep {epoch:3d}/{EPOCHS} loss={avg_loss:.4f} val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    print(f"  Best val F1={best_val_f1:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    results = {'seed': seed, 'best_val_f1': round(best_val_f1 * 100, 2)}
    for split_name, loader in test_loaders.items():
        f1, acc, pr, rc = evaluate(model, loader)
        results[split_name] = {
            'f1': round(f1 * 100, 2),
            'acc': round(acc * 100, 2),
            'pr': round(pr * 100, 2),
            'rc': round(rc * 100, 2),
        }
        print(f"  {split_name}: F1={f1*100:.2f}%")

    out = os.path.join(RESULTS_DIR, f'angle_seed{seed}_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out}")
    return results


def aggregate(all_results):
    conditions = list(SPLITS.keys())
    agg = {}
    for cond in conditions:
        f1s  = [r[cond]['f1']  for r in all_results if cond in r]
        accs = [r[cond]['acc'] for r in all_results if cond in r]
        prs  = [r[cond]['pr']  for r in all_results if cond in r]
        rcs  = [r[cond]['rc']  for r in all_results if cond in r]
        agg[cond] = {
            'f1_mean': round(float(np.mean(f1s)), 2),
            'f1_std':  round(float(np.std(f1s)), 2),
            'acc_mean': round(float(np.mean(accs)), 2),
            'pr_mean':  round(float(np.mean(prs)), 2),
            'rc_mean':  round(float(np.mean(rcs)), 2),
            'all_f1': f1s,
        }

    base_f1 = agg['test']['f1_mean']
    for cond in conditions[1:]:
        agg[cond]['delta_f1'] = round(agg[cond]['f1_mean'] - base_f1, 2)

    agg['n_seeds'] = len(all_results)
    agg['seeds']   = [r['seed'] for r in all_results]
    agg['model']   = 'ANGLE'
    return agg


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading W2V embeddings...")
    vocab, embed_dim, vocab_size, pretrained = load_w2v()
    print(f"  Vocab: {vocab_size}, Embed dim: {embed_dim}")

    print("\nLoading splits (cached after first run)...")
    t0 = time.time()
    train_graphs = load_split('train')
    valid_graphs = load_split('valid')
    test_graphs  = {k: load_split(v) for k, v in SPLITS.items()}
    print(f"Loaded in {time.time()-t0:.1f}s: train={len(train_graphs)}, valid={len(valid_graphs)}")

    all_results = []
    for seed in SEEDS:
        r = train_one_seed(seed, train_graphs, valid_graphs, test_graphs,
                           vocab, embed_dim, vocab_size, pretrained)
        all_results.append(r)

    agg = aggregate(all_results)
    out_path = os.path.join(RESULTS_DIR, 'angle_multiseed_results.json')
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*60}")
    print("ANGLE Multi-seed Summary")
    print(f"{'='*60}")
    for cond, m in agg.items():
        if isinstance(m, dict):
            delta = f"  ΔF1={m.get('delta_f1', 0):+.2f}" if cond != 'test' else ''
            print(f"  {cond:<30} F1={m['f1_mean']:.2f} ± {m['f1_std']:.2f}%{delta}")
    print(f"\nResults → {out_path}")


if __name__ == '__main__':
    main()
