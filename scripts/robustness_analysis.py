#!/usr/bin/env python3
"""
robustness_analysis.py

Three analyses supporting the claim that identifier renaming has low effect:

  1. Token distribution shift    — quantify how much tokens actually change
  2. Embedding similarity        — cosine sim of CodeBERT [CLS] before/after
  3. GNN feature identity        — node feature vectors unchanged by construction

Output: ~/thesis/devign_full/robustness_analysis.json  +  printed summary
"""

import json, sys, gc
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

DEVIGN_DIR  = Path.home() / "GNN-ReGVD/dataset_devign"
RESULT_DIR  = Path.home() / "thesis/devign_full"
CKPT        = Path.home() / "thesis/baselines/codebert/ckpts_multiseed/codebert_seed42.pt"
CPG_ROOT    = Path.home() / "vul-LMGGNN/data/cpg"
REPO_ROOT   = Path.home() / "vul-LMGGNN"
MAX_FUNCS   = 500   # use first N functions (enough for stable stats, keeps runtime short)
MAX_LENGTH  = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def load_jsonl(path, n=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
            if n and len(rows) >= n:
                break
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token distribution shift
# ─────────────────────────────────────────────────────────────────────────────

def token_distribution_shift():
    print("\n" + "="*60, flush=True)
    print("1. Token Distribution Shift Analysis", flush=True)
    print("="*60, flush=True)

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    orig_rows = load_jsonl(DEVIGN_DIR / "test.jsonl",              MAX_FUNCS)
    ren_rows  = load_jsonl(DEVIGN_DIR / "test_obf_identifier.jsonl", MAX_FUNCS)
    n = min(len(orig_rows), len(ren_rows))
    print(f"Analyzing {n} function pairs...", flush=True)

    token_changed_fracs = []   # fraction of tokens that differ per function
    vocab_orig_all, vocab_ren_all = Counter(), Counter()
    total_tokens_orig, total_tokens_ren = 0, 0
    total_changed = 0

    for i, (orig, ren) in enumerate(zip(orig_rows[:n], ren_rows[:n])):
        toks_o = tokenizer.tokenize(orig["func"])[:MAX_LENGTH]
        toks_r = tokenizer.tokenize(ren["func"])[:MAX_LENGTH]

        vocab_orig_all.update(toks_o)
        vocab_ren_all.update(toks_r)
        total_tokens_orig += len(toks_o)
        total_tokens_ren  += len(toks_r)

        # token-level diff on the aligned prefix
        min_len = min(len(toks_o), len(toks_r))
        changed = sum(a != b for a, b in zip(toks_o[:min_len], toks_r[:min_len]))
        changed += abs(len(toks_o) - len(toks_r))  # length diff counts as changed
        denom = max(len(toks_o), len(toks_r))
        token_changed_fracs.append(changed / denom if denom else 0)
        total_changed += changed

    vocab_orig = set(vocab_orig_all.keys())
    vocab_ren  = set(vocab_ren_all.keys())
    jaccard    = len(vocab_orig & vocab_ren) / len(vocab_orig | vocab_ren)
    new_tokens = vocab_ren - vocab_orig   # tokens only in renamed

    result = {
        "n_functions":           n,
        "total_tokens_orig":     total_tokens_orig,
        "total_tokens_renamed":  total_tokens_ren,
        "pct_tokens_changed_mean": round(float(np.mean(token_changed_fracs)) * 100, 2),
        "pct_tokens_changed_std":  round(float(np.std(token_changed_fracs))  * 100, 2),
        "vocab_size_orig":       len(vocab_orig),
        "vocab_size_renamed":    len(vocab_ren),
        "vocab_jaccard_overlap": round(jaccard, 4),
        "new_tokens_introduced": len(new_tokens),
    }

    print(f"  Functions:               {n}", flush=True)
    print(f"  Avg tokens changed/fn:   {result['pct_tokens_changed_mean']:.1f}% ± {result['pct_tokens_changed_std']:.1f}%", flush=True)
    print(f"  Vocabulary (orig):       {result['vocab_size_orig']} unique tokens", flush=True)
    print(f"  Vocabulary (renamed):    {result['vocab_size_renamed']} unique tokens", flush=True)
    print(f"  Vocab Jaccard overlap:   {result['vocab_jaccard_overlap']:.4f}", flush=True)
    print(f"  New tokens introduced:   {result['new_tokens_introduced']}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding similarity (fine-tuned CodeBERT [CLS])
# ─────────────────────────────────────────────────────────────────────────────

def embedding_similarity():
    print("\n" + "="*60, flush=True)
    print("2. CodeBERT Embedding Similarity (fine-tuned, seed 42)", flush=True)
    print("="*60, flush=True)

    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model     = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2)
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded checkpoint: {CKPT.name}", flush=True)

    orig_rows = load_jsonl(DEVIGN_DIR / "test.jsonl",                MAX_FUNCS)
    ren_rows  = load_jsonl(DEVIGN_DIR / "test_obf_identifier.jsonl", MAX_FUNCS)
    n = min(len(orig_rows), len(ren_rows))
    print(f"  Processing {n} pairs...", flush=True)

    cosine_sims = []
    l2_dists    = []
    BATCH = 16

    def encode(texts):
        enc = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model.roberta(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            )
        return out.last_hidden_state[:, 0, :]   # [CLS] token

    for i in range(0, n, BATCH):
        batch_o = [orig_rows[j]["func"] for j in range(i, min(i+BATCH, n))]
        batch_r = [ren_rows[j]["func"]  for j in range(i, min(i+BATCH, n))]
        emb_o = encode(batch_o)
        emb_r = encode(batch_r)
        sims  = F.cosine_similarity(emb_o, emb_r, dim=1).cpu().tolist()
        dists = (emb_o - emb_r).norm(dim=1).cpu().tolist()
        cosine_sims.extend(sims)
        l2_dists.extend(dists)
        print(f"  Batch {i//BATCH+1}/{(n+BATCH-1)//BATCH}  "
              f"mean_cos={np.mean(cosine_sims):.4f}", end="\r", flush=True)

    print(flush=True)
    result = {
        "n_pairs":          n,
        "cosine_sim_mean":  round(float(np.mean(cosine_sims)),  4),
        "cosine_sim_std":   round(float(np.std(cosine_sims)),   4),
        "cosine_sim_min":   round(float(np.min(cosine_sims)),   4),
        "cosine_sim_p5":    round(float(np.percentile(cosine_sims,  5)), 4),
        "cosine_sim_p25":   round(float(np.percentile(cosine_sims, 25)), 4),
        "cosine_sim_median":round(float(np.median(cosine_sims)),4),
        "l2_dist_mean":     round(float(np.mean(l2_dists)),     4),
        "l2_dist_std":      round(float(np.std(l2_dists)),      4),
    }

    print(f"  Cosine similarity:  {result['cosine_sim_mean']:.4f} ± {result['cosine_sim_std']:.4f}", flush=True)
    print(f"  Median:  {result['cosine_sim_median']:.4f}   P5: {result['cosine_sim_p5']:.4f}   P25: {result['cosine_sim_p25']:.4f}", flush=True)
    print(f"  L2 distance:  {result['l2_dist_mean']:.4f} ± {result['l2_dist_std']:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. GNN feature identity (CPG node features unchanged by renaming)
# ─────────────────────────────────────────────────────────────────────────────

def gnn_feature_identity():
    print("\n" + "="*60, flush=True)
    print("3. GNN Feature Identity Check (CPG node features)", flush=True)
    print("="*60, flush=True)

    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    _sys.path.insert(0, str(REPO_ROOT / "models"))
    import importlib.util, pandas as pd
    from collections import OrderedDict
    from torch_geometric.data import Data

    spec = importlib.util.spec_from_file_location(
        "cpg_func2", str(REPO_ROOT / "utils/functions/cpg.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    NODES_DIM    = 205
    FEAT_DIM     = 101
    DEFAULT_TYPE = 65

    def to_feature_vec(cpg_dict):
        try:
            nodes = mod.parse_to_nodes(cpg_dict, max_nodes=NODES_DIM)
        except Exception:
            nodes = OrderedDict()
        x = torch.zeros(NODES_DIM, dtype=torch.float)
        for i, (nid, node) in enumerate(nodes.items()):
            if i >= NODES_DIM: break
            x[i] = float(node.type if node.type is not None else DEFAULT_TYPE)
        return x

    orig_pkls = sorted((CPG_ROOT / "test").glob("*.pkl"),
                       key=lambda p: int(p.stem.split("_")[0]))
    ren_pkls  = sorted((CPG_ROOT / "test_obf_identifier").glob("*.pkl"),
                       key=lambda p: int(p.stem.split("_")[0]))
    n_chunks = min(len(orig_pkls), len(ren_pkls), 5)  # first 5 chunks ~1000 fns

    identical = 0
    total     = 0
    max_diff  = 0.0
    l2_diffs  = []

    for ci in range(n_chunks):
        df_o = pd.read_pickle(orig_pkls[ci])
        df_r = pd.read_pickle(ren_pkls[ci])
        shared = df_o.index.intersection(df_r.index)
        for idx in shared:
            x_o = to_feature_vec(df_o.loc[idx, "cpg"])
            x_r = to_feature_vec(df_r.loc[idx, "cpg"])
            diff = (x_o - x_r).abs().max().item()
            l2   = (x_o - x_r).norm().item()
            l2_diffs.append(l2)
            max_diff = max(max_diff, diff)
            if diff == 0.0:
                identical += 1
            total += 1
        del df_o, df_r
        print(f"  Chunk {ci+1}/{n_chunks}: {total} pairs, "
              f"{identical} identical ({100*identical/total:.1f}%)", flush=True)

    result = {
        "n_chunks_checked":     n_chunks,
        "n_function_pairs":     total,
        "pct_feature_identical":round(100 * identical / total, 2) if total else 0,
        "max_feature_diff":     round(max_diff, 6),
        "l2_diff_mean":         round(float(np.mean(l2_diffs)), 6),
    }

    print(f"\n  Feature vectors identical:  {result['pct_feature_identical']:.1f}% "
          f"of {total} pairs", flush=True)
    print(f"  Max element-wise diff:      {result['max_feature_diff']}", flush=True)
    print(f"  Mean L2 diff:               {result['l2_diff_mean']}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    results["token_shift"]        = token_distribution_shift()
    results["embedding_similarity"]= embedding_similarity()
    results["gnn_feature_identity"]= gnn_feature_identity()

    out = RESULT_DIR / "robustness_analysis.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    ts = results["token_shift"]
    es = results["embedding_similarity"]
    gf = results["gnn_feature_identity"]
    print(f"Token shift:    {ts['pct_tokens_changed_mean']:.1f}% of tokens change "
          f"(Jaccard vocab overlap: {ts['vocab_jaccard_overlap']:.3f})", flush=True)
    print(f"Embedding sim:  cosine = {es['cosine_sim_mean']:.4f} ± {es['cosine_sim_std']:.4f} "
          f"(median {es['cosine_sim_median']:.4f})", flush=True)
    print(f"GNN features:   {gf['pct_feature_identical']:.1f}% identical "
          f"(max diff = {gf['max_feature_diff']})", flush=True)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
