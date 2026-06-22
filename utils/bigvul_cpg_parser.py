"""
CPG parser for Big-Vul — same logic as baselines/vulgnn/cpg_parser.py
but reads from ~/bigvul_cpg/ instead of ~/vul-LMGGNN/data/cpg/.
"""
import os, sys
import glob
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "vulgnn"))
from cpg_parser import (
    cpg_to_graph, tokenize_code,
    NODE_TYPES, NODE2INT, NUM_NODE_TYPES,
    EDGE_TYPES, EDGE2INT, NUM_EDGE_TYPES,
)

BIGVUL_CPG_ROOT = os.path.expanduser("~/bigvul_cpg")


def load_split(split: str, node_feats: str = 'type', max_chunks: int = None) -> list:
    split_dir = os.path.join(BIGVUL_CPG_ROOT, split)
    pkl_files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files in {split_dir}")
    if max_chunks:
        pkl_files = pkl_files[:max_chunks]

    graphs = []
    for pkl_path in pkl_files:
        df = pd.read_pickle(pkl_path)
        for _, row in df.iterrows():
            g = cpg_to_graph(row["cpg"], row["func"], int(row["target"]),
                             node_feats=node_feats)
            if g is not None:
                graphs.append(g)
    return graphs
