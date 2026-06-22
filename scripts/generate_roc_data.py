#!/usr/bin/env python3
"""
generate_roc_data.py — Extract softmax probability scores for ROC curve generation.

Runs inference on the Devign clean test set for all 11 models using seed 42.
Saves probs + labels to ~/thesis/devign_full/roc_data.json.

Usage:
    CUDA_VISIBLE_DEVICES=1 /home/jesse/venvs/reveal310/bin/python3 ~/thesis/generate_roc_data.py
"""

import json, os, sys, glob, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import roc_auc_score

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THESIS_ROOT = Path(__file__).resolve().parent
CPG_ROOT    = Path.home() / "vul-LMGGNN/data/cpg"
AFTER_GGNN  = THESIS_ROOT / "devign_full/after_ggnn"
DEVIGN_ROOT = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_ROOT / "devign_full_split_801010.json"
DATA_FILE   = DEVIGN_ROOT / "originals_full_data_with_slices.json"
OUT_FILE    = DEVIGN_ROOT / "roc_data.json"
BATCH       = 128

print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# CPG data loading for ECG RGCN / VulGNN / ReGVD / Vul-LMGGNN
# (uses VulGNN's cpg_parser with node_feats='type')
# ─────────────────────────────────────────────────────────────
def load_cpg_split_type(split_name):
    sys.path.insert(0, str(Path.home() / "vulgnn_devign"))
    from cpg_parser import load_split as _load
    graphs = _load(split_name, node_feats='type')
    sys.path.pop(0)
    return graphs

# ─────────────────────────────────────────────────────────────
# Helper: get probs from GNN model outputting [B,2] logits
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_probs_pyg(model, loader):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(batch.y.long().cpu().numpy().tolist())
    return np.array(probs), np.array(labels)

# ─────────────────────────────────────────────────────────────
# ECG RGCN
# ─────────────────────────────────────────────────────────────
def extract_ecg_rgcn(graphs):
    # Import here so sys.path is set correctly
    sys.path.insert(0, str(Path.home() / "vulgnn_devign"))
    from cpg_parser import NUM_NODE_TYPES, NUM_EDGE_TYPES
    sys.path.pop(0)

    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
    ECG_HIDDEN = 128; ECG_LAYERS = 3; ECG_EMBED = 32; ECG_DROP = 0.1

    class ECGRGCNDevign(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, ECG_EMBED, padding_idx=0)
            self.input_proj = nn.Linear(ECG_EMBED, ECG_HIDDEN)
            self.convs      = nn.ModuleList([
                RGCNConv(ECG_HIDDEN, ECG_HIDDEN, num_relations=NUM_EDGE_TYPES)
                for _ in range(ECG_LAYERS)])
            self.norms      = nn.ModuleList([nn.LayerNorm(ECG_HIDDEN) for _ in range(ECG_LAYERS)])
            self.drop       = nn.Dropout(ECG_DROP)
            # Sequential to match saved checkpoint (keys 0 and 3)
            self.classifier = nn.Sequential(
                nn.Linear(ECG_HIDDEN * 2, ECG_HIDDEN),
                nn.ReLU(),
                nn.Dropout(ECG_DROP),
                nn.Linear(ECG_HIDDEN, 2),
            )

        def forward(self, data):
            x, ei, et, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if x.dim() > 1: x = x.squeeze(-1)
            x = F.relu(self.input_proj(self.embed(x.long())))
            for conv, norm in zip(self.convs, self.norms):
                x = norm(F.relu(conv(x, ei, et)) + x)
                x = self.drop(x)
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(g)

    ckpt = Path.home() / "ecgrgcn_devign_ckpts/ecgrgcn_dv_seed42.pt"
    model = ECGRGCNDevign().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    loader = PyGLoader(graphs, batch_size=BATCH, shuffle=False)
    return extract_probs_pyg(model, loader)

# ─────────────────────────────────────────────────────────────
# ANGLE  (token-vocabulary model; needs its own data pipeline)
# ─────────────────────────────────────────────────────────────
def extract_angle():
    ANGLE_DIR = Path.home() / "angle_devign"
    sys.path.insert(0, str(ANGLE_DIR))
    from model import ANGLE as ANGLEModel
    from cpg_parser import load_split as angle_load
    from gensim.models import KeyedVectors
    import json as _json

    vocab = _json.load(open(ANGLE_DIR / "vocab.json"))
    wv    = KeyedVectors.load(str(ANGLE_DIR / "w2v_model.bin"))
    vocab_size = len(vocab)
    embed_dim  = wv.vector_size
    emb_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in wv:
            emb_matrix[idx - 2] = wv[word]
    pretrained_emb = torch.from_numpy(emb_matrix)

    MAX_SEQ_LEN = 30
    unk = vocab.get('<unk>', 1)

    import re as _re

    def encode_node_tokens(codes, vocab, max_seq):
        result = []
        for code in codes:
            toks = _re.split(r'\W+', code.lower())
            toks = [t for t in toks if t]
            ids  = [vocab.get(t, unk) for t in toks][:max_seq]
            if len(ids) < max_seq:
                ids += [0] * (max_seq - len(ids))
            result.append(ids)
        return torch.tensor(result, dtype=torch.long)

    def graphs_to_token_data(raw_graphs, vocab):
        from torch_geometric.data import Data
        token_graphs = []
        for g in raw_graphs:
            codes = g.node_codes if hasattr(g, 'node_codes') else [''] * g.num_nodes
            x_tok = encode_node_tokens(codes, vocab, MAX_SEQ_LEN)
            d = Data(x=x_tok, edge_index=g.edge_index,
                     edge_attr=g.edge_attr, y=g.y)
            token_graphs.append(d)
        return token_graphs

    raw_test  = angle_load('test', node_feats='code')
    test_graphs = graphs_to_token_data(raw_test, vocab)
    sys.path.pop(0)
    # Remove cached module so next import of 'model' gets vulgnn's version
    sys.modules.pop('model', None)
    sys.modules.pop('cpg_parser', None)

    ckpt = ANGLE_DIR / "checkpoints/angle_seed42.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = ANGLEModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden=64,
        pool_ratio=0.5,
        num_layers=3,
        dropout=0.1,
        pretrained_emb=pretrained_emb,
    ).to(DEVICE)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"])

    loader = PyGLoader(test_graphs, batch_size=BATCH, shuffle=False)
    return extract_probs_pyg(model, loader)

# ─────────────────────────────────────────────────────────────
# VulGNN
# ─────────────────────────────────────────────────────────────
def extract_vulgnn(graphs):
    # Clear any angle_devign model.py from sys.path first
    sys.path = [p for p in sys.path if 'angle_devign' not in p]
    sys.path.insert(0, str(Path.home() / "vulgnn_devign"))
    # Remove cached module so Python re-imports from the new path
    sys.modules.pop('model', None)

    from model import VulGNN as VulGNNModel
    ckpt = Path.home() / "vulgnn_devign/checkpoints/vulgnn_seed42.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    # state is a plain OrderedDict (no wrapper)
    model = VulGNNModel(
        num_node_types=14, num_edge_types=2,
        embed_dim=16, edge_dim=4, hidden=128, num_layers=6, dropout=0.08
    ).to(DEVICE)
    model.load_state_dict(state)
    loader = PyGLoader(graphs, batch_size=BATCH, shuffle=False)
    return extract_probs_pyg(model, loader)

# ─────────────────────────────────────────────────────────────
# REVEAL (sklearn LR on GGNN embeddings — clean and reproducible)
# ─────────────────────────────────────────────────────────────
def extract_reveal():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def load_emb(path):
        data = json.load(open(path))
        X = np.array([d["graph_feature"] for d in data], dtype=np.float32)
        Y = np.array([d["target"] for d in data], dtype=np.int64)
        return X, Y

    X_train, Y_train = load_emb(AFTER_GGNN / "train_GGNNinput_graph.json")
    X_valid, Y_valid = load_emb(AFTER_GGNN / "valid_GGNNinput_graph.json")
    X_test,  Y_test  = load_emb(AFTER_GGNN / "test_GGNNinput_graph.json")
    X_pool = np.concatenate([X_train, X_valid])
    Y_pool = np.concatenate([Y_train, Y_valid])

    scaler = StandardScaler().fit(X_pool)
    X_pool_s = scaler.transform(X_pool)
    X_test_s = scaler.transform(X_test)

    # Average over 10 seeds for stability
    all_probs = []
    for seed in range(10):
        np.random.seed(1000 + seed * 7)
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed,
                                 class_weight='balanced', solver='lbfgs')
        clf.fit(X_pool_s, Y_pool)
        p = clf.predict_proba(X_test_s)[:, 1]
        all_probs.append(p)
    return np.mean(all_probs, axis=0), Y_test

# ─────────────────────────────────────────────────────────────
# ReGVD
# ─────────────────────────────────────────────────────────────
def extract_regvd(graphs):
    sys.path.insert(0, str(THESIS_ROOT / "baselines/regvd"))
    from train_regvd import ReGVDModel
    from transformers import RobertaTokenizer, RobertaModel

    ckpt_path = THESIS_ROOT / "baselines/regvd/models/regvd_devign/best.pt"
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = state.get("config", {})

    tokenizer  = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model = ReGVDModel(bert_model, cfg).to(DEVICE)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state,
                          strict=False)
    model.eval()

    loader = PyGLoader(graphs, batch_size=32, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch.y.long().cpu().numpy().tolist())
    return np.array(probs), np.array(labels)

# ─────────────────────────────────────────────────────────────
# Vul-LMGGNN
# ─────────────────────────────────────────────────────────────
def extract_lmggnn(graphs):
    lmggnn_dir = str(Path.home() / "vul-LMGGNN")
    sys.path.insert(0, lmggnn_dir)
    sys.path.insert(0, lmggnn_dir + "/models")  # for 'layers' import inside LMGNN.py
    from models.LMGNN import BertGGCN
    import json as _json
    cfg_path = Path.home() / "vul-LMGGNN/configs.json"
    cfg = _json.load(open(cfg_path))
    model_cfg  = cfg["bertggnn"]["model"]
    gated_args = model_cfg["gated_graph_conv_args"]
    conv_args  = model_cfg["conv_args"]
    emb_size   = model_cfg["emb_size"]

    ckpt_path  = Path.home() / "vul-LMGGNN/data/model/lmggnn_seed42.pt"
    model = BertGGCN(gated_args, conv_args, emb_size, DEVICE).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"], strict=False)
    model.eval()

    loader = PyGLoader(graphs, batch_size=16, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch.y.long().cpu().numpy().tolist())
    return np.array(probs), np.array(labels)

# ─────────────────────────────────────────────────────────────
# CodeBERT / CodeT5+ / CodeBERT-Aug
# Uses originals_full_data_with_slices.json + split file
# ─────────────────────────────────────────────────────────────
def extract_transformer(ckpt_path, is_codet5=False):
    from torch.utils.data import Dataset, DataLoader

    # Load full dataset and split
    split  = json.load(open(SPLIT_FILE))
    all_data = json.load(open(DATA_FILE))
    test_names = set(split["splits"]["test"])
    idx = {r["file_name"]: r for r in all_data}
    test_recs = [idx[n] for n in split["splits"]["test"] if n in idx]
    print(f"  {len(test_recs)} test records", flush=True)

    MAX_LEN = 512
    if is_codet5:
        from transformers import AutoTokenizer, T5EncoderModel

        MODEL_NAME = "Salesforce/codet5p-220m"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class CodeT5PlusClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder    = T5EncoderModel.from_pretrained(MODEL_NAME)
                hidden          = self.encoder.config.d_model
                self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 2))

            def forward_logits(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)

        model = CodeT5PlusClassifier()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                              else state["model_state_dict"])
        model = model.to(DEVICE).eval()

        class _DataLoader:
            pass

        loader = torch.utils.data.DataLoader(
            __import__('torch').utils.data.TensorDataset(
                *[torch.zeros(1)]), batch_size=1)  # placeholder - will use direct loop

        probs, labels = [], []
        from torch.utils.data import DataLoader as _DL, Dataset as _DS

        class TxtDS(_DS):
            def __init__(self, recs):
                self.recs = recs
            def __len__(self): return len(self.recs)
            def __getitem__(self, i):
                r = self.recs[i]
                code = r.get("code", r.get("func", ""))[:3000]
                enc = tokenizer(code, max_length=MAX_LEN, padding="max_length",
                               truncation=True, return_tensors="pt")
                return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), \
                       int(r.get("label", r.get("target", 0)))

        dl = _DL(TxtDS(test_recs), batch_size=16, shuffle=False, num_workers=2)
        with torch.no_grad():
            for iids, amsk, lbl in dl:
                logits = model.forward_logits(iids.to(DEVICE), amsk.to(DEVICE))
                p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
                labels.extend(lbl.numpy().tolist())
        return np.array(probs), np.array(labels)
    else:
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", num_labels=2)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model = model.to(DEVICE).eval()

    class TxtDataset(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            code = r.get("code", r.get("func", ""))[:3000]
            enc = tokenizer(code, max_length=MAX_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), \
                   int(r.get("label", r.get("target", 0)))

    loader = DataLoader(TxtDataset(test_recs), batch_size=16, shuffle=False, num_workers=2)
    probs, labels = [], []
    with torch.no_grad():
        for iids, amsk, lbl in loader:
            out = model(input_ids=iids.to(DEVICE), attention_mask=amsk.to(DEVICE))
            p = F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(lbl.numpy().tolist())
    return np.array(probs), np.array(labels)

# ─────────────────────────────────────────────────────────────
# TF-IDF + LR
# ─────────────────────────────────────────────────────────────
def extract_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    split    = json.load(open(SPLIT_FILE))
    all_data = json.load(open(DATA_FILE))
    idx = {r["file_name"]: r for r in all_data}

    tr_recs = [idx[n] for n in split["splits"]["train"] if n in idx]
    te_recs = [idx[n] for n in split["splits"]["test"]  if n in idx]
    tr_txt = [r.get("code", r.get("func", "")) for r in tr_recs]
    te_txt = [r.get("code", r.get("func", "")) for r in te_recs]
    tr_lbl = [int(r.get("label", r.get("target", 0))) for r in tr_recs]
    te_lbl = [int(r.get("label", r.get("target", 0))) for r in te_recs]

    vec = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_tr = vec.fit_transform(tr_txt)
    X_te = vec.transform(te_txt)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
    clf.fit(X_tr, tr_lbl)
    probs = clf.predict_proba(X_te)[:, 1]
    return probs, np.array(te_lbl)

# ─────────────────────────────────────────────────────────────
# CPG + LR
# ─────────────────────────────────────────────────────────────
def extract_cpglr():
    from sklearn.linear_model import LogisticRegression

    def load_cpg_features(split_name):
        split_dir = CPG_ROOT / split_name
        pkl_files = sorted(glob.glob(str(split_dir / "*.pkl")))
        feats, labels = [], []
        for pkl_path in pkl_files:
            df = pd.read_pickle(pkl_path)
            for _, row in df.iterrows():
                cpg = row["cpg"]
                n_nodes = len(cpg.get("nodes", []))
                edges   = cpg.get("edges", [])
                n_edges = len(edges)
                etypes  = len(set(e.get("etype", 0) for e in edges)) if edges else 0
                feats.append([n_nodes, n_edges, etypes,
                              n_edges / max(n_nodes, 1),
                              len(str(row.get("func","")).split())])
                labels.append(int(row["target"]))
        return np.array(feats, dtype=np.float32), np.array(labels)

    X_tr, Y_tr = load_cpg_features("train")
    X_te, Y_te = load_cpg_features("test")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_tr, Y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    return probs, Y_te

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # Load existing partial results so we can skip completed models
    roc_data = {}
    if OUT_FILE.exists():
        roc_data = json.load(open(OUT_FILE))
        print(f"Loaded existing partial results: {list(roc_data.keys())}")

    # Load CPG type-encoded graphs once (ECG RGCN, VulGNN, ReGVD, Vul-LMGGNN)
    print("\nLoading Devign test CPG graphs (type encoding)...", flush=True)
    cpg_graphs = load_cpg_split_type("test")
    print(f"  {len(cpg_graphs)} graphs loaded", flush=True)

    tasks = [
        ("ECG RGCN",     lambda: extract_ecg_rgcn(cpg_graphs)),
        ("ANGLE",        lambda: extract_angle()),
        ("VulGNN",       lambda: extract_vulgnn(cpg_graphs)),
        ("REVEAL",       lambda: extract_reveal()),
        ("ReGVD",        lambda: extract_regvd(cpg_graphs)),
        ("Vul-LMGGNN",   lambda: extract_lmggnn(cpg_graphs)),
        ("CodeBERT",     lambda: extract_transformer(
            THESIS_ROOT/"baselines/codebert/ckpts_multiseed/codebert_seed42.pt")),
        ("CodeT5+",      lambda: extract_transformer(
            THESIS_ROOT/"baselines/codebert/ckpts_codet5plus/codet5plus_seed42.pt",
            is_codet5=True)),
        ("CodeBERT-Aug", lambda: extract_transformer(
            THESIS_ROOT/"baselines/codebert/ckpts_augmented/codebert_aug_devign_seed42.pt")),
        ("TF-IDF+LR",    lambda: extract_tfidf()),
        ("CPG+LR",       lambda: extract_cpglr()),
    ]

    for name, fn in tasks:
        if name in roc_data:
            print(f"\n  Skipping {name} (already done, AUC={roc_data[name]['auc']})", flush=True)
            continue
        print(f"\n{'='*50}\n  {name}\n{'='*50}", flush=True)
        try:
            probs, labels = fn()
            auc = roc_auc_score(labels, probs)
            print(f"  AUC-ROC = {auc:.4f}  n={len(labels)}", flush=True)
            roc_data[name] = {"probs": probs.tolist(), "labels": labels.tolist(),
                              "auc": round(auc, 4)}
            # Save incrementally
            with open(OUT_FILE, "w") as f:
                json.dump(roc_data, f)
            print(f"  Saved to {OUT_FILE}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\nDone. Models extracted: {list(roc_data.keys())}")
    print(f"Results → {OUT_FILE}")


if __name__ == "__main__":
    main()
