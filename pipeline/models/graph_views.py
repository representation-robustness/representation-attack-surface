"""Turn exported Joern graphs into model inputs ("views").

A view fixes three things, set per model in configs/graph_models.yaml:
  level     : "stmt" = CFG nodes only (REVEAL/Devign statement graphs), "ast" = every AST node
  edges     : subset of AST, CFG, CDG, REACHING_DEF (edge_type ids follow this order)
  features  : "type"      structural node type only (renaming-invariant by construction)
              "type+w2v"  type plus mean Word2Vec of the node's code tokens (lexical channel)

Node type = Joern label, refined by operator name for CALL nodes and by the
keyword for CONTROL_STRUCTURE; vocabulary and Word2Vec are fit on the clean
training split only, so identifiers introduced by a transformation are OOV.
Built views are cached per (dataset, split, condition, view) under $RERUN_ROOT/cache/views.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from pipeline.models.common import ROOT, load_graphs

EDGE_ORDER = ["AST", "CFG", "CDG", "REACHING_DEF"]
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|&&|\|\||<<|>>|\+\+|--|[^\sA-Za-z0-9_]")
W2V_DIM = 100
MAX_TYPES = 200


def node_type(label: str, code: str, name: str) -> str:
    if label == "CALL" and name and name.startswith("<operator"):
        return "OP:" + name
    if label == "CONTROL_STRUCTURE":
        return "CS:" + (code.split("(")[0].split("{")[0].strip().split(" ")[0] if code else "")
    return label


def tokens(code: str) -> list[str]:
    return TOKEN.findall(code or "")


class Vocab:
    """Node-type vocabulary + Word2Vec, fit once per dataset on clean train graphs."""

    def __init__(self, dataset: str):
        self.dir = ROOT / "cache" / "vocab" / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        tp, wp = self.dir / "types.json", self.dir / "w2v.model"
        # One fit per dataset even when several jobs start together: Word2Vec with several workers is not
        # bit-reproducible, so concurrent fits would give jobs different embeddings. Others wait on the lock.
        with open(self.dir / ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not tp.exists() or not wp.exists():
                self._fit(dataset, tp, wp)
        self.types = {t: i for i, t in enumerate(json.loads(tp.read_text()))}
        from gensim.models import Word2Vec
        self.w2v = Word2Vec.load(str(wp)).wv

    @staticmethod
    def _fit(dataset, tp, wp):
        from gensim.models import Word2Vec
        cnt, sents = Counter(), []
        for g in load_graphs(dataset, "train", "clean"):
            for _, label, code, name, *_ in g["nodes"]:
                cnt[node_type(label, code, name)] += 1
                if label != "METHOD":  # METHOD code is the whole function
                    toks = tokens(code)
                    if toks:
                        sents.append(toks)
        types = ["<UNK>"] + [t for t, _ in cnt.most_common(MAX_TYPES - 1)]
        tp.write_text(json.dumps(types))
        Word2Vec(sents, vector_size=W2V_DIM, window=5, min_count=3, workers=3, seed=1, epochs=5).save(str(wp))

    @property
    def n_types(self) -> int:
        return len(self.types)

    def embed(self, code: str) -> np.ndarray:
        vs = [self.w2v[t] for t in tokens(code) if t in self.w2v]
        return np.mean(vs, 0) if vs else np.zeros(W2V_DIM, dtype=np.float32)


def to_data(g: dict, vocab: Vocab, level: str, edges: list[str], features: str) -> Data | None:
    nodes = g["nodes"]
    keep = [n for n in nodes if level == "ast" or n[7] == 1]
    if not keep:
        return None
    idx = {n[0]: i for i, n in enumerate(keep)}
    etypes = {e: k for k, e in enumerate(edges)}
    src, dst, et = [], [], []
    for s, d, t in g["edges"]:
        if t in etypes and s in idx and d in idx:
            src.append(idx[s]); dst.append(idx[d]); et.append(etypes[t])
    x_type = torch.tensor([vocab.types.get(node_type(n[1], n[2], n[3]), 0) for n in keep], dtype=torch.long)
    d = Data(x_type=x_type, num_nodes=len(keep),
             edge_index=torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long),
             edge_type=torch.tensor(et, dtype=torch.long),
             y=torch.tensor([g["target"]], dtype=torch.long))
    if features == "type+w2v":
        d.x_w2v = torch.from_numpy(np.stack([vocab.embed(n[2]) for n in keep]).astype(np.float16))
    d.sid = g["id"]
    return d


def load_view(dataset: str, split: str, condition: str, view: dict) -> list[Data]:
    key = hashlib.sha1(json.dumps(view, sort_keys=True).encode()).hexdigest()[:10]
    path = ROOT / "cache" / "views" / dataset / f"{split}_{condition}_{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent jobs (several seeds start together) share one cache file: one builds it while the others wait on the
    # lock, and the file appears atomically (tmp + rename), so no job ever reads a partially written file.
    with open(path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if path.exists():
            return torch.load(path, weights_only=False)
        vocab = Vocab(dataset)
        out = [d for g in load_graphs(dataset, split, condition)
               if (d := to_data(g, vocab, view["level"], view["edges"], view["features"])) is not None]
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(out, tmp)
        os.replace(tmp, path)
        (path.with_suffix(".json")).write_text(json.dumps(view))
        return out


def node_features(batch, n_types: int, features: str) -> torch.Tensor:
    """Dense node features for models taking a float matrix: one-hot type [+ w2v]."""
    x = torch.nn.functional.one_hot(batch.x_type, n_types).float()
    if features == "type+w2v":
        x = torch.cat([x, batch.x_w2v.float()], dim=1)
    return x
