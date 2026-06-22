"""
CPG parser for DiverseVul — reads from ~/diversevul_cpg/ instead of ~/bigvul_cpg/.
"""
import os, sys
import glob
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "vulgnn"))
from cpg_parser import (
    cpg_to_graph, tokenize_code,
    NODE_TYPES, NODE2INT, NUM_NODE_TYPES,
    EDGE_TYPES, EDGE2INT, NUM_EDGE_TYPES,
)

DIVERSEVUL_CPG_ROOT = os.path.expanduser("~/diversevul_cpg")


def load_split(split: str, node_feats: str = "type", max_chunks: int = None) -> list:
    split_dir = os.path.join(DIVERSEVUL_CPG_ROOT, split)
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
