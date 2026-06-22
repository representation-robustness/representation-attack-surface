#!/usr/bin/env python3
"""
generate_roc_data_diversevul.py — Extract softmax probability scores for DiverseVul ROC curves.

Usage:
    CUDA_VISIBLE_DEVICES=2 /home/jesse/venvs/reveal310/bin/python3 ~/thesis/generate_roc_data_diversevul.py
"""

import json, os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import roc_auc_score

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THESIS_ROOT = Path(__file__).resolve().parent
DV_SPLITS   = THESIS_ROOT / "diversevul_dataset/splits"
LMGGNN_DIR  = str(Path.home() / "vul-LMGGNN")
OUT_FILE    = THESIS_ROOT / "devign_full/roc_data_diversevul.json"
BATCH       = 128

print(f"Device: {DEVICE}")


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


def extract_ecg_rgcn(graphs):
    sys.path.insert(0, str(Path.home()))
    from diversevul_cpg_parser import NUM_NODE_TYPES, NUM_EDGE_TYPES
    sys.path.pop(0)

    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
    HIDDEN = 128; LAYERS = 3; EMBED = 32; DROP = 0.1

    class ECGRGCNDiverseVul(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, EMBED, padding_idx=0)
            self.input_proj = nn.Linear(EMBED, HIDDEN)
            self.convs      = nn.ModuleList([
                RGCNConv(HIDDEN, HIDDEN, num_relations=NUM_EDGE_TYPES)
                for _ in range(LAYERS)])
            self.norms      = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
            self.dropout    = nn.Dropout(DROP)
            self.classifier = nn.Sequential(
                nn.Linear(HIDDEN * 2, HIDDEN), nn.ReLU(),
                nn.Dropout(DROP), nn.Linear(HIDDEN, 2))

        def forward(self, data):
            x, ei, et, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if x.dim() > 1: x = x.squeeze(-1)
            x = F.relu(self.input_proj(self.embed(x.long())))
            for conv, norm in zip(self.convs, self.norms):
                x = norm(F.relu(conv(x, ei, et)) + x)
                x = self.dropout(x)
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(g)

    ckpt = Path.home() / "ecgrgcn_diversevul_ckpts/ecgrgcn_dv_seed42.pt"
    model = ECGRGCNDiverseVul().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    return extract_probs_pyg(model, PyGLoader(graphs, batch_size=BATCH, shuffle=False))


def extract_angle(cpg_code_graphs):
    ANGLE_DIR = Path.home() / "angle_devign"
    sys.path.insert(0, str(ANGLE_DIR))
    from model import ANGLE as ANGLEModel
    import json as _json

    vocab = _json.load(open(ANGLE_DIR / "vocab.json"))
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load(str(ANGLE_DIR / "w2v_model.bin"))
    vocab_size = len(vocab); embed_dim = wv.vector_size
    emb_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for word, idx in vocab.items():
        if word in wv:
            emb_matrix[idx - 2] = wv[word]
    pretrained_emb = torch.from_numpy(emb_matrix)

    MAX_SEQ_LEN = 16
    unk = vocab.get('<unk>', 1)
    import re as _re

    def encode_node_tokens(codes, vocab, max_seq):
        result = []
        for code in codes:
            toks = _re.split(r'\W+', code.lower())
            toks = [t for t in toks if t]
            ids = [vocab.get(t, unk) for t in toks][:max_seq]
            if len(ids) < max_seq:
                ids += [0] * (max_seq - len(ids))
            result.append(ids)
        return torch.tensor(result, dtype=torch.long)

    from torch_geometric.data import Data
    token_graphs = []
    for g in cpg_code_graphs:
        codes = g.node_codes if hasattr(g, 'node_codes') else [''] * g.num_nodes
        x_tok = encode_node_tokens(codes, vocab, MAX_SEQ_LEN)
        token_graphs.append(Data(x=x_tok, edge_index=g.edge_index,
                                 edge_attr=g.edge_attr, y=g.y))

    sys.path.pop(0)
    sys.modules.pop('model', None)
    sys.modules.pop('cpg_parser', None)

    ckpt = Path.home() / "angle_diversevul_ckpts/angle_dv_seed42.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = ANGLEModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden=64,
                       pool_ratio=0.5, num_layers=3, dropout=0.1,
                       pretrained_emb=pretrained_emb).to(DEVICE)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"])
    return extract_probs_pyg(model, PyGLoader(token_graphs, batch_size=BATCH, shuffle=False))


def extract_vulgnn(graphs):
    sys.path = [p for p in sys.path if 'angle_devign' not in p]
    sys.path.insert(0, str(Path.home() / "vulgnn_devign"))
    sys.modules.pop('model', None)

    from model import VulGNN as VulGNNModel
    ckpt = Path.home() / "vulgnn_diversevul_ckpts/vulgnn_dv_seed42.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = VulGNNModel(num_node_types=14, num_edge_types=2,
                        embed_dim=16, edge_dim=4, hidden=128,
                        num_layers=6, dropout=0.08).to(DEVICE)
    model.load_state_dict(state)
    return extract_probs_pyg(model, PyGLoader(graphs, batch_size=BATCH, shuffle=False))


def extract_reveal(graphs):
    from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
    HIDDEN = 200; NUM_BLOCKS = 4; STEPS = 2; EMBED = 32; DROP = 0.3

    sys.path.insert(0, str(Path.home()))
    from diversevul_cpg_parser import NUM_NODE_TYPES
    sys.path.pop(0)

    class REVEALDiverseVul(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed      = nn.Embedding(NUM_NODE_TYPES + 1, EMBED, padding_idx=0)
            self.input_proj = nn.Linear(EMBED, HIDDEN)
            self.blocks     = nn.ModuleList([
                nn.ModuleList([GatedGraphConv(HIDDEN, STEPS), nn.LayerNorm(HIDDEN)])
                for _ in range(NUM_BLOCKS)])
            self.dropout    = nn.Dropout(DROP)
            self.classifier = nn.Sequential(
                nn.Linear(HIDDEN * 2, HIDDEN), nn.ReLU(),
                nn.Dropout(DROP), nn.Linear(HIDDEN, 2))

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x.dim() > 1: x = x.squeeze(-1)
            x = F.relu(self.input_proj(self.embed(x.long())))
            for gru, norm in self.blocks:
                res = x
                x = gru(x, edge_index)
                x = norm(x + res)
                x = F.relu(x); x = self.dropout(x)
            return self.classifier(torch.cat([global_mean_pool(x, batch),
                                              global_max_pool(x, batch)], dim=-1))

    ckpt = Path.home() / "reveal_sys_diversevul_fixed_ckpts/reveal_dv_fixed_seed42.pt"
    model = REVEALDiverseVul().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    return extract_probs_pyg(model, PyGLoader(graphs, batch_size=BATCH, shuffle=False))


def extract_lmggnn(graphs):
    sys.path.insert(0, LMGGNN_DIR)
    sys.path.insert(0, LMGGNN_DIR + "/models")
    from models.LMGNN import BertGGCN
    import json as _json
    cfg = _json.load(open(Path.home() / "vul-LMGGNN/configs.json"))
    model_cfg  = cfg["bertggnn"]["model"]
    gated_args = model_cfg["gated_graph_conv_args"]
    conv_args  = model_cfg["conv_args"]
    emb_size   = model_cfg["emb_size"]

    ckpt_path = Path.home() / "vul-LMGGNN/data/model/lmggnn_diversevul_seed42.pt"
    model = BertGGCN(gated_args, conv_args, emb_size, DEVICE).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"], strict=False)
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for batch in PyGLoader(graphs, batch_size=16, shuffle=False):
            batch = batch.to(DEVICE)
            logits = model(batch)
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(batch.y.long().cpu().numpy().tolist())
    return np.array(probs), np.array(labels)


def load_jsonl_dv(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    return [{"code": r["func"][:3000], "label": int(r["target"])} for r in rows]


def extract_codebert(ckpt_path):
    from torch.utils.data import Dataset, DataLoader
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    test_recs = load_jsonl_dv(DV_SPLITS / "test.jsonl")
    print(f"  {len(test_recs)} test records", flush=True)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    class TxtDS(Dataset):
        def __init__(self, recs): self.recs = recs
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            enc = tokenizer(r["code"], max_length=512, padding="max_length",
                            truncation=True, return_tensors="pt")
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), r["label"]

    probs, labels = [], []
    with torch.no_grad():
        for iids, amsk, lbl in DataLoader(TxtDS(test_recs), batch_size=16, shuffle=False, num_workers=2):
            out = model(input_ids=iids.to(DEVICE), attention_mask=amsk.to(DEVICE))
            p = F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(lbl.numpy().tolist())
    return np.array(probs), np.array(labels)


def extract_codet5(ckpt_path):
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, T5EncoderModel

    test_recs = load_jsonl_dv(DV_SPLITS / "test.jsonl")
    print(f"  {len(test_recs)} test records", flush=True)
    MODEL_NAME = "Salesforce/codet5p-220m"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    class TxtDS(Dataset):
        def __init__(self, recs): self.recs = recs
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            enc = tokenizer(r["code"], max_length=512, padding="max_length",
                            truncation=True, return_tensors="pt")
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), r["label"]

    probs, labels = [], []
    with torch.no_grad():
        for iids, amsk, lbl in DataLoader(TxtDS(test_recs), batch_size=16, shuffle=False, num_workers=2):
            logits = model.forward_logits(iids.to(DEVICE), amsk.to(DEVICE))
            p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(lbl.numpy().tolist())
    return np.array(probs), np.array(labels)


def extract_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr_recs = load_jsonl_dv(DV_SPLITS / "train.jsonl")
    te_recs = load_jsonl_dv(DV_SPLITS / "test.jsonl")
    tr_txt = [r["code"] for r in tr_recs]; te_txt = [r["code"] for r in te_recs]
    tr_lbl = [r["label"] for r in tr_recs]; te_lbl = [r["label"] for r in te_recs]

    vec = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_tr = vec.fit_transform(tr_txt)
    X_te = vec.transform(te_txt)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced')
    clf.fit(X_tr, tr_lbl)
    probs = clf.predict_proba(X_te)[:, 1]
    return probs, np.array(te_lbl)


def main():
    roc_data = {}
    if OUT_FILE.exists():
        roc_data = json.load(open(OUT_FILE))
        print(f"Loaded existing partial results: {list(roc_data.keys())}")

    print("\nLoading DiverseVul CPG test graphs...", flush=True)
    sys.path.insert(0, str(Path.home()))
    from diversevul_cpg_parser import load_split as dv_load
    cpg_graphs      = dv_load('test')
    cpg_code_graphs = dv_load('test', node_feats='code')
    print(f"  {len(cpg_graphs)} graphs loaded", flush=True)

    CB_CKPT  = THESIS_ROOT / "baselines/codebert/ckpts_diversevul_codebert/codebert_dv_seed42.pt"
    CT5_CKPT = THESIS_ROOT / "baselines/codebert/ckpts_diversevul_codet5plus/codet5plus_dv_seed42.pt"

    tasks = [
        ("ECG RGCN",   lambda: extract_ecg_rgcn(cpg_graphs)),
        ("ANGLE",      lambda: extract_angle(cpg_code_graphs)),
        ("VulGNN",     lambda: extract_vulgnn(cpg_graphs)),
        ("REVEAL",     lambda: extract_reveal(cpg_graphs)),
        ("Vul-LMGGNN", lambda: extract_lmggnn(cpg_graphs)),
        ("CodeBERT",   lambda: extract_codebert(CB_CKPT)),
        ("CodeT5+",    lambda: extract_codet5(CT5_CKPT)),
        ("TF-IDF+LR",  lambda: extract_tfidf()),
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
            with open(OUT_FILE, "w") as f:
                json.dump(roc_data, f)
            print(f"  Saved to {OUT_FILE}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\nDone. Models: {list(roc_data.keys())}")
    print(f"Results → {OUT_FILE}")


if __name__ == "__main__":
    main()
