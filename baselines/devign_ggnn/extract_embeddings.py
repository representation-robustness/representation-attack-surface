#!/usr/bin/env python3
"""
Extract graph-level GGNN embeddings from a trained DevignModel checkpoint.

Supports two modes:
  --from_pickle: Load DataSet from processed.bin (fast, for originals)
  --input JSON:  Build graphs from a raw GGNN input JSON (for obf test sets)

Output after_ggnn format: [{"graph_feature": [...], "target": 0/1}, ...]

Usage:
    # Originals (train/valid/test from pickle):
    python extract_embeddings.py --from_pickle PATH/processed.bin \
        --split all --checkpoint ... --output_dir PATH/after_ggnn/

    # Obf test set from JSON:
    python extract_embeddings.py --input PATH/test_GGNNinput.json \
        --checkpoint ... --output PATH/test_GGNNinput_graph.json
"""
import argparse
import json
import os
import pickle
import sys
import torch
from tqdm import tqdm
from data_loader.batch_graph import GGNNBatchGraph
from modules.model import DevignModel
from utils import initialize_batch, debug


def build_single_batch(examples):
    bg = GGNNBatchGraph()
    for e in examples:
        bg.add_subgraph(e.graph)
    return bg


def extract_from_examples(model, examples, device, batch_size=64):
    """Extract mean-pooled GGNN embeddings for a list of DataEntry objects."""
    model.eval()
    all_embs = []
    all_labels = []
    # initialize_batch returns batches of *indices* into examples
    batches = initialize_batch(examples, batch_size, shuffle=False)
    with torch.no_grad():
        for batch_indices in tqdm(batches, desc='Extracting', file=sys.stderr):
            batch_examples = [examples[int(i)] for i in batch_indices]
            bg = build_single_batch(batch_examples)
            graph, features, edge_types = bg.get_network_inputs(
                cuda=(device.type != 'cpu'), device=device
            )
            node_outputs = model.ggnn(graph, features, edge_types)
            h_i, lengths = bg.de_batchify_graphs(node_outputs)  # [B, max_N, D]
            for i, le in enumerate(lengths):
                vec = h_i[i, :le.item(), :].mean(dim=0).cpu().tolist()
                all_embs.append(vec)
            all_labels.extend(e.target for e in batch_examples)
    return all_embs, all_labels


def examples_from_json(json_path, edge_type_registry=None):
    """Build DataEntry-like objects from a raw GGNN input JSON.
    edge_type_registry: dict mapping edge_type_int -> index; built in-place if None.
    """
    from data_loader.dataset import DataEntry
    import dgl

    class _DS:
        def __init__(self, registry):
            self.edge_types = registry or {}
            self.max_etype = max(self.edge_types.values(), default=-1) + 1

        def get_edge_type_number(self, t):
            if t not in self.edge_types:
                self.edge_types[t] = self.max_etype
                self.max_etype += 1
            return self.edge_types[t]

    ds = _DS(edge_type_registry)
    debug(f'Loading {json_path}')
    with open(json_path) as f:
        raw = json.load(f)
    examples = []
    for entry in tqdm(raw, desc='Building graphs', file=sys.stderr):
        e = DataEntry(
            datset=ds,
            num_nodes=len(entry['node_features']),
            features=entry['node_features'],
            edges=entry['graph'],
            target=entry['targets'][0][0],
        )
        examples.append(e)
    debug(f'  Built {len(examples)} graphs; edge types: {ds.max_etype}')
    return examples, ds.edge_types


def load_model(checkpoint_path, input_dim, output_dim, max_edge_types, num_steps, device):
    model = DevignModel(
        input_dim=input_dim,
        output_dim=output_dim,
        max_edge_types=max_edge_types,
        num_steps=num_steps,
    )
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.to(device)
    debug(f'Loaded checkpoint (val_f1={ck.get("best_val_f1", "?")})')
    return model


def save_after_ggnn(embs, labels, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = [{'graph_feature': e, 'target': int(l)} for e, l in zip(embs, labels)]
    with open(path, 'w') as f:
        json.dump(out, f)
    debug(f'Saved {len(out)} embeddings → {path}  (dim={len(out[0]["graph_feature"])})')
    return len(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature_size', type=int, default=169)
    parser.add_argument('--graph_embed_size', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--from_pickle', metavar='PICKLE',
                      help='Load DataSet from processed.bin (originals)')
    mode.add_argument('--input', metavar='JSON',
                      help='Raw GGNN input JSON (single split, e.g. obf test)')

    parser.add_argument('--output', metavar='PATH',
                        help='Output path (required with --input)')
    parser.add_argument('--output_dir', metavar='DIR',
                        help='Output directory (required with --from_pickle)')
    parser.add_argument('--split', default='all',
                        choices=['train', 'valid', 'test', 'all'],
                        help='Which split to extract (with --from_pickle)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug(f'Device: {device}')

    if args.from_pickle:
        assert args.output_dir, '--output_dir required with --from_pickle'
        debug(f'Loading pickle: {args.from_pickle}')
        dataset = pickle.load(open(args.from_pickle, 'rb'))
        fs = dataset.feature_size
        met = dataset.max_edge_type
        debug(f'  feature_size={fs}, max_edge_type={met}')
        model = load_model(args.checkpoint, fs, args.graph_embed_size, met,
                           args.num_steps, device)
        splits = {
            'train': dataset.train_examples,
            'valid': dataset.valid_examples,
            'test':  dataset.test_examples,
        }
        targets = ['train', 'valid', 'test'] if args.split == 'all' else [args.split]
        for split in targets:
            examples = splits[split]
            debug(f'Extracting {split}: {len(examples)} examples')
            embs, labels = extract_from_examples(model, examples, device, args.batch_size)
            out_path = os.path.join(args.output_dir, f'{split}_GGNNinput_graph.json')
            save_after_ggnn(embs, labels, out_path)

    else:
        assert args.output, '--output required with --input'
        examples, edge_registry = examples_from_json(args.input)
        met = max(edge_registry.values(), default=0) + 1
        model = load_model(args.checkpoint, args.feature_size, args.graph_embed_size,
                           met, args.num_steps, device)
        debug(f'Extracting {len(examples)} examples')
        embs, labels = extract_from_examples(model, examples, device, args.batch_size)
        save_after_ggnn(embs, labels, args.output)

    print('Done.')


if __name__ == '__main__':
    main()
