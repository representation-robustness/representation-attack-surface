"""
train_w2v.py — Train Word2Vec on node code tokens from train CPGs.

Saves:
    ~/angle_devign/w2v_model.bin   — gensim KeyedVectors
    ~/angle_devign/vocab.json      — token → index mapping

Usage:
    python train_w2v.py
"""

import os
import sys
import json
import time
from collections import Counter

from gensim.models import Word2Vec

sys.path.insert(0, os.path.dirname(__file__))
from cpg_parser import load_split, tokenize_code

_DIR       = os.path.dirname(os.path.abspath(__file__))
W2V_PATH   = os.path.join(_DIR, 'w2v_model.bin')
VOCAB_PATH = os.path.join(_DIR, 'vocab.json')
EMB_DIM    = 100
MIN_COUNT  = 2
EPOCHS_W2V = 10
WORKERS    = 4


def collect_sentences(splits=('train',)):
    """Collect token sequences from all node code in given splits."""
    sentences = []
    for split in splits:
        print(f"  Collecting tokens from {split}...")
        graphs = load_split(split, node_feats='code')
        for g in graphs:
            codes = getattr(g, 'node_codes', [])
            for code in codes:
                toks = tokenize_code(code)
                if toks:
                    sentences.append(toks)
    return sentences


def main():
    print("=== Training Word2Vec on CPG node code tokens ===")

    print("Loading CPGs (train + valid)...")
    t0 = time.time()
    sentences = collect_sentences(splits=['train', 'valid'])
    print(f"  {len(sentences)} token sequences in {time.time()-t0:.1f}s")

    # Train Word2Vec
    print(f"\nTraining Word2Vec (dim={EMB_DIM}, epochs={EPOCHS_W2V})...")
    t0 = time.time()
    model = Word2Vec(
        sentences,
        vector_size=EMB_DIM,
        window=5,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS_W2V,
        sg=1,  # skip-gram
    )
    model.wv.save(W2V_PATH)
    print(f"  Vocab size: {len(model.wv):,}  |  time: {time.time()-t0:.1f}s")
    print(f"  Saved → {W2V_PATH}")

    # Build token → index vocab (0 = PAD, 1..N = tokens)
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(model.wv.index_to_key, start=2):
        vocab[word] = i
    with open(VOCAB_PATH, 'w') as f:
        json.dump(vocab, f)
    print(f"  Vocab saved → {VOCAB_PATH}  ({len(vocab)} entries)")


if __name__ == '__main__':
    main()
