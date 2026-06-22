#!/usr/bin/env python3
"""
Extract graph-level embeddings from the trained Devign GGNN for all data splits.

For each function graph, runs the GGNN and computes a length-normalised mean
of the final node states.  Saves results in the after_ggnn format expected by
REVEAL's representation_learning/api_test.py:

    [{"graph_feature": [float, ...], "target": 0_or_1}, ...]

Usage (from baselines/devign/):
    python extract_ggnn_embeddings.py
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parents[1]
DEVIGN_INPUT = THESIS_ROOT / "devign_full" / "devign_input"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "models" / "devign_full_originals" / "best_model.pt"

SPLITS = {
    # name → (input_dir, ggnn_prefix, out_filename)
    "originals_train": (DEVIGN_INPUT / "originals_train", "train_GGNNinput.json", "train_GGNNinput_graph.json"),
    "originals_valid": (DEVIGN_INPUT / "originals_train", "valid_GGNNinput.json", "valid_GGNNinput_graph.json"),
    "originals_test":  (DEVIGN_INPUT / "originals_train", "test_GGNNinput.json",  "test_GGNNinput_graph.json"),
    "obf_identifier":  (DEVIGN_INPUT / "obf_identifier_test", "test_GGNNinput.json", "obf_identifier_test_GGNNinput_graph.json"),
    "obf_deadcode":    (DEVIGN_INPUT / "obf_deadcode_test",   "test_GGNNinput.json", "obf_deadcode_test_GGNNinput_graph.json"),
    "obf_controlflow": (DEVIGN_INPUT / "obf_controlflow_test","test_GGNNinput.json", "obf_controlflow_test_GGNNinput_graph.json"),
}

# ---------------------------------------------------------------------------
# Devign model / data-loader imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
from modules.model import DevignModel
from data_loader.dataset import DataEntry, DataSet
from data_loader.batch_graph import GGNNBatchGraph
from utils import initialize_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: torch.device) -> DevignModel:
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    # Infer dims from saved weights
    output_dim = sd["ggnn.linears.0.weight"].shape[0]   # 200
    concat_dim = sd["mlp_z.weight"].shape[1]            # 369
    input_dim  = concat_dim - output_dim                # 169
    max_edge_types = len(set(k.split(".")[2] for k in sd if k.startswith("ggnn.linears.")))

    model = DevignModel(
        input_dim=input_dim,
        output_dim=output_dim,
        max_edge_types=max_edge_types,
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Loaded model  input_dim={input_dim}  output_dim={output_dim}  "
          f"max_edge_types={max_edge_types}", flush=True)
    return model, output_dim


def read_raw_json(path: Path):
    """Return list of raw GGNN-input dicts."""
    with open(str(path)) as f:
        return json.load(f)


def make_entries(raw_list, edge_types: dict, max_etype_ref: list):
    """Convert raw dicts to DataEntry objects (shared edge_types dict)."""
    entries = []
    for rec in raw_list:
        node_features = rec["node_features"]
        edges         = rec["graph"]          # list of [src, type, dst]
        target        = int(rec["targets"][0][0])

        # Inline edge-type registration (mirrors DataSet.get_edge_type_number)
        graph_edges = []
        for s, etype, t in edges:
            if etype not in edge_types:
                edge_types[etype] = max_etype_ref[0]
                max_etype_ref[0] += 1
            graph_edges.append((s, edge_types[etype], t))

        entries.append((node_features, graph_edges, target))
    return entries


def extract_embeddings(entries, model: DevignModel, output_dim: int,
                       device: torch.device, batch_size: int = 64):
    """
    Run GGNN on each entry; return list of (embedding_list, target) tuples.
    Embedding = length-normalised mean of final node states (shape [output_dim]).
    """
    from dgl import DGLGraph

    results = []
    # Process one graph at a time to avoid padding issues across very different sizes
    for node_features, graph_edges, target in tqdm(entries):
        try:
            nf = torch.FloatTensor(node_features).to(device)
            g  = DGLGraph()
            g.add_nodes(len(node_features), data={"features": nf})
            if graph_edges:
                src = torch.LongTensor([e[0] for e in graph_edges])
                dst = torch.LongTensor([e[2] for e in graph_edges])
                et  = torch.LongTensor([e[1] for e in graph_edges])
                g.add_edges(src, dst, data={"etype": et})

            with torch.no_grad():
                feats = g.ndata["features"].to(device)
                etypes = g.edata["etype"].to(device) if g.num_edges() > 0 else torch.zeros(0, dtype=torch.long, device=device)
                # Run DGL GatedGraphConv directly
                node_out = model.ggnn(g, feats, etypes)   # [num_nodes, output_dim]
                embedding = node_out.mean(dim=0)           # [output_dim]

            results.append((embedding.cpu().tolist(), target))
        except Exception as exc:
            print(f"  Skipped graph ({len(node_features)} nodes): {exc}", file=sys.stderr)
            # Use zero vector as fallback so record count stays consistent
            results.append(([0.0] * output_dim, target))

    return results


def save_after_ggnn(embeddings, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"graph_feature": emb, "target": tgt} for emb, tgt in embeddings]
    with open(str(out_path), "w") as f:
        json.dump(records, f)
    pos = sum(r["target"] for r in records)
    print(f"  Saved {len(records)} records  pos={pos}  neg={len(records)-pos}  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for after_ggnn files (default: devign_full/after_ggnn)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else THESIS_ROOT / "devign_full" / "after_ggnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")   # no GPU needed for inference
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")

    model, output_dim = load_model(checkpoint_path, device)

    # Shared edge-type registry so train/valid/test/obf all use consistent indices
    edge_types: dict = {}
    max_etype_ref = [0]

    # We need to pre-populate edge_types from the training data first so that
    # test/obf splits don't see new edge types as index 0.
    # (In practice the edge-type vocabulary is usually {0,1,2,3} = AST/CFG/CDG/DFG)

    summary = {}
    for split_name, (src_dir, in_file, out_file) in SPLITS.items():
        in_path  = src_dir / in_file
        out_path = out_dir / out_file
        print(f"\n[{split_name}]  {in_path}")

        raw = read_raw_json(in_path)
        print(f"  Loaded {len(raw)} graphs")

        entries = make_entries(raw, edge_types, max_etype_ref)
        embeddings = extract_embeddings(entries, model, output_dim, device)
        save_after_ggnn(embeddings, out_path)
        summary[split_name] = {"records": len(embeddings), "path": str(out_path)}

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v['records']} records → {v['path']}")

    # Save summary JSON
    summary_path = out_dir / "extraction_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
