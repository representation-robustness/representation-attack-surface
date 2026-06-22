"""
train.py — Train and evaluate ANGLE on Devign.

Usage:
    CUDA_VISIBLE_DEVICES=3 python train.py

Requires:
    ~/angle_devign/w2v_model.bin  (run train_w2v.py first)
    ~/angle_devign/vocab.json

Outputs:
    ~/angle_devign/angle_devign.pt
    ~/thesis/devign_full/angle_results.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, tokenize_code
from model import ANGLE

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY= 1e-5
BATCH       = 64
HIDDEN      = 64
NUM_LAYERS  = 3
POOL_RATIO  = 0.5
DROPOUT     = 0.1
MAX_SEQ_LEN = 16      # max tokens per node

_DIR        = os.path.dirname(os.path.abspath(__file__))
W2V_PATH    = os.path.join(_DIR, 'w2v_model.bin')
VOCAB_PATH  = os.path.join(_DIR, 'vocab.json')
CKPT_PATH   = os.path.join(_DIR, 'angle_devign.pt')
RESULTS_OUT = os.path.expanduser('~/thesis/devign_full/angle_results.json')

SPLITS = {
    'test':                 'test',
    'test_obf_identifier':  'test_obf_identifier',
    'test_obf_deadcode':    'test_obf_deadcode',
    'test_obf_controlflow': 'test_obf_controlflow',
}


def load_vocab_and_w2v():
    """Load vocabulary and Word2Vec weights."""
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    wv = KeyedVectors.load(W2V_PATH)
    vocab_size = len(vocab)
    embed_dim  = wv.vector_size

    # Build embedding matrix: index 0=pad, 1=unk, 2..N = vocab tokens
    emb_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if idx < 2:
            continue
        if word in wv:
            emb_matrix[idx - 2] = wv[word]   # offset -2 since ANGLE.embed starts at 2

    return vocab, torch.from_numpy(emb_matrix)


def encode_node_tokens(codes: list, vocab: dict, max_seq: int) -> torch.LongTensor:
    """
    Encode list of node code strings → LongTensor [N, max_seq].
    Each row is the token indices for that node's code, padded to max_seq.
    """
    rows = []
    unk = vocab.get('<unk>', 1)
    for code in codes:
        toks = tokenize_code(code)[:max_seq]
        ids  = [vocab.get(t, unk) for t in toks]
        # Pad to max_seq
        ids += [0] * (max_seq - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)   # [N, max_seq]


def graphs_to_token_data(graphs_code: list, vocab: dict) -> list:
    """
    Convert graphs with node_codes → graphs with x as token index tensors.
    """
    token_graphs = []
    for g in graphs_code:
        codes = getattr(g, 'node_codes', [''] * g.num_nodes)
        x_tok = encode_node_tokens(codes, vocab, MAX_SEQ_LEN)  # [N, MAX_SEQ_LEN]
        d = Data(
            x=x_tok,
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            y=g.y,
        )
        d.func = g.func
        token_graphs.append(d)
    return token_graphs


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
    return f1, acc


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


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load vocab + Word2Vec ─────────────────────────────────────────────────
    print("\nLoading vocab and Word2Vec...")
    vocab, pretrained_emb = load_vocab_and_w2v()
    vocab_size = len(vocab)
    embed_dim  = pretrained_emb.shape[1]
    print(f"  vocab_size={vocab_size}, embed_dim={embed_dim}")

    # ── Load CPGs with code ───────────────────────────────────────────────────
    print("\nLoading train split (with code)...")
    t0 = time.time()
    train_raw = load_split('train', node_feats='code')
    print(f"  {len(train_raw)} graphs in {time.time()-t0:.1f}s")

    print("Encoding node tokens...")
    train_graphs = graphs_to_token_data(train_raw, vocab)
    del train_raw

    print("Loading valid split...")
    valid_raw    = load_split('valid', node_feats='code')
    valid_graphs = graphs_to_token_data(valid_raw, vocab)
    del valid_raw

    print("Loading test splits...")
    test_graphs = {}
    for k, v in SPLITS.items():
        raw = load_split(v, node_feats='code')
        test_graphs[k] = graphs_to_token_data(raw, vocab)
        print(f"  {k}: {len(test_graphs[k])} graphs")

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=BATCH)
    test_loaders = {k: DataLoader(v, batch_size=BATCH) for k, v in test_graphs.items()}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ANGLE(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden=HIDDEN,
        pool_ratio=POOL_RATIO,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_emb=pretrained_emb,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nANGLE parameters: {n_params:,}")

    weights   = compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = 0

    print(f"\nTraining ANGLE for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

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

        val_f1, val_acc = evaluate(model, valid_loader)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"val_F1={val_f1:.4f} | val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  *** new best (val_F1={val_f1:.4f}) ***")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\nBest val F1={best_val_f1:.4f} at epoch {best_epoch}")
    print("Loading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    results = {}
    for split_name, loader in test_loaders.items():
        f1, acc = evaluate(model, loader)
        results[split_name] = {'f1': round(f1 * 100, 2), 'accuracy': round(acc * 100, 2)}
        print(f"  {split_name}: F1={f1*100:.2f}%  acc={acc*100:.2f}%")

    results['best_val_f1'] = round(best_val_f1 * 100, 2)
    results['best_epoch']  = best_epoch
    results['model'] = 'ANGLE'

    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_OUT}")


if __name__ == '__main__':
    main()
