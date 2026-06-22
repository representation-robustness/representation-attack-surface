#!/usr/bin/env python3
"""
Exp 2: Distribution shift diagnostics for dead-code insertion.

Computes for each dataset (Devign, BigVul, DiverseVul):
  1. Token JS divergence (TF-IDF, clean vs dead-code test set)
  2. Vocab shift        (OOV/rare token rate change)
  3. Pattern frequency  (dead-code template tokens in vulnerable training fns)
  4. Graph stat divergence (node/edge type distributions from CPG data)

Output: devign_full/distribution_shift_results.json
"""

import json, pickle, os, re, sys
import numpy as np
from pathlib import Path
from collections import Counter

THESIS   = Path(__file__).resolve().parents[1]
DEVIGN   = THESIS / "devign_full"
BIGVUL   = Path("/home/jesse/bigvul_cpg")
DIVVUL   = Path("/home/jesse/diversevul_cpg")
OUT      = DEVIGN / "distribution_shift_results.json"

DEAD_TEMPLATE_TOKENS = {
    "__dc_", "__fl_", "__dm_", "__deadcode__", "(v ^ v)", "(v | v)"
}

def tokenize(text):
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[^\s\w]', text or "")

def js_divergence(p_counts, q_counts):
    """Jensen-Shannon divergence between two token count dicts."""
    vocab = set(p_counts) | set(q_counts)
    p_total = max(sum(p_counts.values()), 1)
    q_total = max(sum(q_counts.values()), 1)
    p = np.array([p_counts.get(v, 0) / p_total for v in vocab])
    q = np.array([q_counts.get(v, 0) / q_total for v in vocab])
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 6)

def oov_rate(texts, vocab):
    total, oov = 0, 0
    for t in texts:
        for tok in tokenize(t):
            total += 1
            if tok not in vocab:
                oov += 1
    return oov / max(total, 1)

def pattern_freq(texts, labels, templates=DEAD_TEMPLATE_TOKENS):
    """Fraction of vulnerable training functions containing any template token."""
    vuln_texts = [t for t, l in zip(texts, labels) if l == 1]
    if not vuln_texts:
        return 0.0
    hits = sum(1 for t in vuln_texts if any(tp in t for tp in templates))
    return round(hits / len(vuln_texts), 6)

def load_json_split(json_path, split_path, split="test"):
    data = json.loads(Path(json_path).read_text())
    idx  = json.loads(Path(split_path).read_text())
    idx  = idx.get("splits", idx)
    test_files = set(idx.get(split, []))
    return [r for r in data if r.get("file_name") in test_files]

def load_json_split_train(json_path, split_path):
    data = json.loads(Path(json_path).read_text())
    idx  = json.loads(Path(split_path).read_text())
    idx  = idx.get("splits", idx)
    train_files = set(idx.get("train", []))
    return [r for r in data if r.get("file_name") in train_files]

def load_pkl_dir(directory):
    recs = []
    for f in sorted(Path(directory).glob("*.pkl")):
        df = pickle.load(open(f, "rb"))
        recs.extend(df.to_dict("records"))
    return recs

def graph_divergence_from_ggnn(clean_path, obf_path):
    """Compare node-type and edge-type frequency distributions."""
    def count_types(path):
        node_c, edge_c, sizes = Counter(), Counter(), []
        try:
            graphs = json.loads(Path(path).read_text())
            if isinstance(graphs, dict):
                graphs = graphs.get("graphs", [])
            for g in graphs:
                nf = g.get("node_features", [])
                sizes.append(len(nf))
                for feat in nf:
                    if isinstance(feat, list):
                        node_c[tuple(feat[:3])] += 1
                for src, dst, etype in g.get("edges", []):
                    edge_c[etype] += 1
        except Exception:
            pass
        return node_c, edge_c, sizes
    nc, ec, sz   = count_types(clean_path)
    no, eo, szo  = count_types(obf_path)
    node_jsd = js_divergence(nc, no)
    edge_jsd = js_divergence(ec, eo)
    size_delta = round(np.mean(szo) / max(np.mean(sz), 1) - 1, 4) if sz and szo else None
    return {"node_type_jsd": node_jsd, "edge_type_jsd": edge_jsd, "size_ratio_delta": size_delta}

results = {}

# ── DEVIGN ───────────────────────────────────────────────────────────────────
print("Processing Devign ...", flush=True)
split_file  = DEVIGN / "devign_full_split_801010.json"
orig_file   = DEVIGN / "originals_full_data_with_slices.json"
dead_file   = DEVIGN / "obf_deadcode_full_data_with_slices.json"

try:
    orig_test   = load_json_split(orig_file, split_file, "test")
    dead_test   = load_json_split(dead_file, split_file, "test")
    train_recs  = load_json_split_train(orig_file, split_file)

    orig_texts  = [r["code"] for r in orig_test]
    dead_texts  = [r["code"] for r in dead_test]
    train_texts = [r["code"] for r in train_recs]
    train_labels= [r["label"] for r in train_recs]

    orig_counts = Counter(tok for t in orig_texts for tok in tokenize(t))
    dead_counts = Counter(tok for t in dead_texts for tok in tokenize(t))
    train_vocab = set(orig_counts.keys())

    token_jsd   = js_divergence(orig_counts, dead_counts)
    oov_clean   = oov_rate(orig_texts, train_vocab)
    oov_dead    = oov_rate(dead_texts, train_vocab)
    pat_freq    = pattern_freq(train_texts, train_labels)

    # Graph divergence from GGNN JSON
    clean_ggnn = DEVIGN / "devign_input" / "originals_train" / "test_GGNNinput.json"
    dead_ggnn  = DEVIGN / "devign_input" / "obf_deadcode_test" / "test_GGNNinput.json"
    graph_div  = graph_divergence_from_ggnn(clean_ggnn, dead_ggnn)

    results["devign"] = {
        "token_jsd":      round(token_jsd, 6),
        "oov_clean":      round(oov_clean, 6),
        "oov_dead":       round(oov_dead, 6),
        "vocab_shift":    round(oov_dead - oov_clean, 6),
        "pattern_freq_train_vuln": pat_freq,
        "graph_divergence": graph_div,
    }
    print(f"  Devign: token_jsd={token_jsd:.6f}, vocab_shift={oov_dead-oov_clean:.6f}", flush=True)
except Exception as e:
    print(f"  Devign ERROR: {e}", flush=True)
    results["devign"] = {"error": str(e)}

# ── BIG-VUL ──────────────────────────────────────────────────────────────────
print("Processing BigVul ...", flush=True)
try:
    orig_bv  = load_pkl_dir(BIGVUL / "test")
    dead_bv  = load_pkl_dir(BIGVUL / "test_obf_deadcode")
    train_bv = load_pkl_dir(BIGVUL / "train")

    orig_texts  = [r["func"] for r in orig_bv]
    dead_texts  = [r["func"] for r in dead_bv]
    train_texts = [r["func"] for r in train_bv]
    train_labels= [r["target"] for r in train_bv]

    orig_counts = Counter(tok for t in orig_texts for tok in tokenize(t))
    dead_counts = Counter(tok for t in dead_texts for tok in tokenize(t))
    train_vocab = set(orig_counts.keys())

    token_jsd = js_divergence(orig_counts, dead_counts)
    oov_clean = oov_rate(orig_texts, train_vocab)
    oov_dead  = oov_rate(dead_texts, train_vocab)
    pat_freq  = pattern_freq(train_texts, train_labels)

    # Graph divergence from CPG node types
    def cpg_node_dist(recs):
        node_c = Counter()
        for r in recs:
            cpg = r.get("cpg", {})
            for func in cpg.get("functions", [{}]):
                for node in func.get("AST", []):
                    nid = node.get("id", "")
                    ntype = nid.split(".")[0] if "." in nid else nid[:20]
                    node_c[ntype] += 1
        return node_c
    nc = cpg_node_dist(orig_bv)
    nd = cpg_node_dist(dead_bv)
    graph_jsd = js_divergence(nc, nd)

    results["bigvul"] = {
        "token_jsd":      round(token_jsd, 6),
        "oov_clean":      round(oov_clean, 6),
        "oov_dead":       round(oov_dead, 6),
        "vocab_shift":    round(oov_dead - oov_clean, 6),
        "pattern_freq_train_vuln": pat_freq,
        "graph_divergence": {"node_type_jsd": round(graph_jsd, 6)},
    }
    print(f"  BigVul: token_jsd={token_jsd:.6f}, vocab_shift={oov_dead-oov_clean:.6f}", flush=True)
except Exception as e:
    print(f"  BigVul ERROR: {e}", flush=True)
    results["bigvul"] = {"error": str(e)}

# ── DIVERSEVUL ───────────────────────────────────────────────────────────────
print("Processing DiverseVul ...", flush=True)
try:
    orig_dv  = load_pkl_dir(DIVVUL / "test")
    dead_dv  = load_pkl_dir(DIVVUL / "test_obf_deadcode")
    train_dv = load_pkl_dir(DIVVUL / "train")

    orig_texts  = [r["func"] for r in orig_dv]
    dead_texts  = [r["func"] for r in dead_dv]
    train_texts = [r["func"] for r in train_dv]
    train_labels= [r["target"] for r in train_dv]

    orig_counts = Counter(tok for t in orig_texts for tok in tokenize(t))
    dead_counts = Counter(tok for t in dead_texts for tok in tokenize(t))
    train_vocab = set(orig_counts.keys())

    token_jsd = js_divergence(orig_counts, dead_counts)
    oov_clean = oov_rate(orig_texts, train_vocab)
    oov_dead  = oov_rate(dead_texts, train_vocab)
    pat_freq  = pattern_freq(train_texts, train_labels)

    def cpg_node_dist(recs):
        node_c = Counter()
        for r in recs:
            cpg = r.get("cpg", {})
            for func in cpg.get("functions", [{}]):
                for node in func.get("AST", []):
                    nid = node.get("id","")
                    ntype = nid.split(".")[0] if "." in nid else nid[:20]
                    node_c[ntype] += 1
        return node_c
    nc = cpg_node_dist(orig_dv)
    nd = cpg_node_dist(dead_dv)
    graph_jsd = js_divergence(nc, nd)

    results["diversevul"] = {
        "token_jsd":      round(token_jsd, 6),
        "oov_clean":      round(oov_clean, 6),
        "oov_dead":       round(oov_dead, 6),
        "vocab_shift":    round(oov_dead - oov_clean, 6),
        "pattern_freq_train_vuln": pat_freq,
        "graph_divergence": {"node_type_jsd": round(graph_jsd, 6)},
    }
    print(f"  DiverseVul: token_jsd={token_jsd:.6f}, vocab_shift={oov_dead-oov_clean:.6f}", flush=True)
except Exception as e:
    print(f"  DiverseVul ERROR: {e}", flush=True)
    results["diversevul"] = {"error": str(e)}

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {OUT}", flush=True)
