#!/usr/bin/env python3
"""Extract per-function predictions for ANGLE on all 7 conditions."""

import json, os, sys, glob
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

# ANGLE-specific imports — must come before any other model imports
ANGLE_DIR = os.path.expanduser("~/angle_devign")
sys.path.insert(0, ANGLE_DIR)

from cpg_parser import load_split
from model import ANGLE

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 256
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

HIDDEN = 64; NUM_LAYERS = 3; POOL_RATIO = 0.5; DROPOUT = 0.1
SEEDS  = [42, 1337, 7, 100, 999]
CKPT_DIR = os.path.join(ANGLE_DIR, "checkpoints")


@torch.no_grad()
def predict_labels(model, loader):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        truths.extend(batch.y.cpu().tolist())
    return preds, truths


def main():
    print(f"Device: {DEVICE}", flush=True)

    # load W2V and vocab
    from gensim.models import KeyedVectors
    vocab = json.load(open(os.path.join(ANGLE_DIR, "vocab.json")))
    wv    = KeyedVectors.load(os.path.join(ANGLE_DIR, "w2v_model.bin"), mmap='r')
    embed_dim  = wv.vector_size
    vocab_size = len(vocab)
    pretrained = torch.zeros(vocab_size + 1, embed_dim)
    for word, idx in vocab.items():
        if word in wv:
            pretrained[idx] = torch.tensor(wv[word])
    print(f"  W2V loaded: vocab={vocab_size} embed_dim={embed_dim}", flush=True)

    print("Loading test splits...", flush=True)
    cond_graphs = {}
    for cond, split in CONDITIONS.items():
        try:
            cond_graphs[cond] = load_split(split)
            print(f"  {cond}: {len(cond_graphs[cond])}", flush=True)
        except Exception as e:
            print(f"  SKIP {cond}: {e}", flush=True)

    true_labels = [int(g.y.item()) for g in cond_graphs["clean"]]
    result = {"model": "angle", "conditions": list(cond_graphs.keys()),
              "true_labels": true_labels, "seeds": {}}

    for seed in SEEDS:
        ckpt = os.path.join(CKPT_DIR, f"angle_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  Seed {seed}: missing, skip", flush=True); continue

        model = ANGLE(vocab_size=vocab_size, embed_dim=embed_dim, hidden=HIDDEN,
                      pool_ratio=POOL_RATIO, num_layers=NUM_LAYERS, dropout=DROPOUT,
                      pretrained_emb=pretrained).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        print(f"\nSeed {seed}", flush=True)

        seed_preds = {}
        for cond, graphs in cond_graphs.items():
            loader = DataLoader(graphs, batch_size=BATCH, shuffle=False)
            preds, _ = predict_labels(model, loader)
            seed_preds[cond] = preds
            n = min(len(true_labels), len(preds))
            print(f"  {cond}: F1={f1_score(true_labels[:n], preds[:n], zero_division=0)*100:.2f}% ({len(preds)} samples)",
                  flush=True)
        result["seeds"][str(seed)] = seed_preds
        del model; torch.cuda.empty_cache()

    out = OUT_DIR / "angle_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
