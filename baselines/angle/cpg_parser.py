"""
cpg_parser.py — Shared CPG pkl → PyG Data conversion for VulGNN and ANGLE.

CPG pkl format (from ~/vul-LMGGNN/data/cpg/{split}/*.pkl):
  columns: func (str), target (int), cpg (dict)
  cpg['functions'][0]['AST'] = list of node dicts:
    { 'id': 'nodes.Call@123',
      'edges': [{'id': 'edges.Ast@src_dst', 'in': 'nodes.X@dst', 'out': 'nodes.Y@src'}, ...],
      'properties': [{'key': 'CODE', 'value': '...'}, ...] }

Node types (14): Call, Identifier, Literal, FieldIdentifier, Block, ControlStructure,
  Local, MethodParameterIn, MethodParameterOut, JumpTarget, Return, Method, MethodReturn, Unknown
Edge types (2): Ast, Cfg
"""

import re
import os
import glob
import pandas as pd
import torch
from torch_geometric.data import Data

# ── Vocabulary ────────────────────────────────────────────────────────────────
NODE_TYPES = [
    'Call', 'Identifier', 'Literal', 'FieldIdentifier', 'Block',
    'ControlStructure', 'Local', 'MethodParameterIn', 'MethodParameterOut',
    'JumpTarget', 'Return', 'Method', 'MethodReturn', 'Unknown'
]
NODE2INT = {t: i for i, t in enumerate(NODE_TYPES)}
NUM_NODE_TYPES = len(NODE_TYPES)

EDGE_TYPES = ['Ast', 'Cfg']
EDGE2INT  = {t: i for i, t in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)

CPG_ROOT = os.path.expanduser("~/vul-LMGGNN/data/cpg")


def _parse_ntype(node_id: str) -> int:
    """Extract node type integer from 'nodes.Call@123'."""
    ntype_str = node_id.split('@')[0].replace('nodes.', '')
    return NODE2INT.get(ntype_str, NODE2INT['Unknown'])


def _parse_etype(edge_id: str) -> int:
    """Extract edge type integer from 'edges.Ast@src_dst'."""
    etype_str = edge_id.split('@')[0].replace('edges.', '')
    return EDGE2INT.get(etype_str, 0)  # default Ast


def _get_code(node: dict) -> str:
    """Extract CODE property from a node dict."""
    for p in node.get('properties', []):
        if p['key'] == 'CODE':
            return p['value'] or ''
    return ''


def cpg_to_graph(cpg_dict: dict, func_text: str, target: int,
                 node_feats: str = 'type') -> Data | None:
    """
    Convert one CPG dict to a PyG Data object.

    Args:
        cpg_dict:   row['cpg'] from pkl
        func_text:  raw source text (func column)
        target:     label (0/1)
        node_feats: 'type'  → x shape [N] (long, type integer)
                    'code'  → x as list of code strings (populated but not tensor)
    Returns:
        PyG Data or None if graph is empty.
    """
    for func in cpg_dict.get('functions', []):
        ast_nodes = func.get('AST', [])
        if not ast_nodes:
            return None

        # Build node index
        node_ids = [n['id'] for n in ast_nodes]
        nid2idx  = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        # Node type integers
        types = [_parse_ntype(n['id']) for n in ast_nodes]

        # Collect edges (deduplicate by edge id)
        seen = set()
        srcs, dsts, etypes = [], [], []
        for node in ast_nodes:
            for e in node.get('edges', []):
                eid = e['id']
                if eid in seen:
                    continue
                seen.add(eid)
                etype_str = eid.split('@')[0].replace('edges.', '')
                if etype_str not in EDGE2INT:
                    continue
                in_nid  = e['in']   # target (Joern inNode)
                out_nid = e['out']  # source (Joern outNode)
                if in_nid not in nid2idx or out_nid not in nid2idx:
                    continue
                srcs.append(nid2idx[out_nid])
                dsts.append(nid2idx[in_nid])
                etypes.append(EDGE2INT[etype_str])

        x_type = torch.tensor(types, dtype=torch.long)
        y      = torch.tensor([target], dtype=torch.long)

        if srcs:
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            edge_attr  = torch.tensor(etypes,        dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros(0,       dtype=torch.long)

        d = Data(x=x_type, edge_index=edge_index, edge_attr=edge_attr, y=y)
        d.func = func_text
        d.num_nodes = N

        # Also attach code strings if requested (for ANGLE Word2Vec)
        if node_feats == 'code':
            d.node_codes = [_get_code(n) for n in ast_nodes]

        return d

    return None


def load_split(split: str, node_feats: str = 'type', max_chunks: int = None) -> list:
    """Load all pkl chunks for a split → list of PyG Data objects."""
    split_dir = os.path.join(CPG_ROOT, split)
    pkls = sorted(glob.glob(os.path.join(split_dir, '*_cpg.pkl')))
    if max_chunks:
        pkls = pkls[:max_chunks]

    graphs = []
    for pkl_path in pkls:
        df = pd.read_pickle(pkl_path)
        for _, row in df.iterrows():
            g = cpg_to_graph(row['cpg'], row['func'], row['target'], node_feats)
            if g is not None:
                graphs.append(g)
    return graphs


def tokenize_code(code: str) -> list:
    """Simple whitespace + camelCase tokenizer for Word2Vec training."""
    # Split on non-alphanumeric, then split camelCase
    tokens = re.split(r'[^a-zA-Z0-9_]', code)
    result = []
    for tok in tokens:
        if not tok:
            continue
        # split camelCase
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', tok).split()
        result.extend(p.lower() for p in parts if p)
    return result or ['<unk>']
