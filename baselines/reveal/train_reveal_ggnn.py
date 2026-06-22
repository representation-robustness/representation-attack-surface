#!/usr/bin/env python3
"""
Step 3 fallback: Train REVEAL's own GatedGraphNeuralNetwork (pure PyTorch, no DGL)
on devign_full data, extract 200-dim graph embeddings, save in after_ggnn format.

Architecture:
  REVEAL GatedGraphNeuralNetwork (hidden_size=200, 8 edge types, 8 timesteps)
  + mean-pool over final node states
  + linear classifier (200 → 1)
  + BCEWithLogitsLoss with pos_weight

Usage (from baselines/reveal_ggnn/):
  CUDA_VISIBLE_DEVICES=6 python train_reveal_ggnn.py
"""
import copy
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parents[1]
REVEAL_ROOT  = THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe"
DEVIGN_INPUT = THESIS_ROOT / "devign_full/devign_input/originals_train"
OBF_DIRS     = {
    "obf_identifier":  THESIS_ROOT / "devign_full/devign_input/obf_identifier_test",
    "obf_deadcode":    THESIS_ROOT / "devign_full/devign_input/obf_deadcode_test",
    "obf_controlflow": THESIS_ROOT / "devign_full/devign_input/obf_controlflow_test",
}
OUT_DIR      = THESIS_ROOT / "devign_full/after_ggnn_reveal"
MODEL_OUT    = SCRIPT_DIR / "models/reveal_ggnn_devign_full"

# ---------------------------------------------------------------------------
# REVEAL GNN imports (pure PyTorch)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(THESIS_ROOT / "data/raw/ReVeal"))
from Vuld_SySe.graph_network.gnn import GatedGraphNeuralNetwork
from Vuld_SySe.graph_network.ggnn_dataset import AdjacencyList

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
HIDDEN_SIZE  = 200
NUM_EDGE_TYPES = 8          # AST/CFG/CDG/DFG and reverses — safe upper bound
LAYER_TIMESTEPS = [8, 8, 8] # 3 layers × 8 steps each
RESIDUAL = {}

class REVEALVulnDetector(nn.Module):
    """REVEAL GGNN + mean-pool + linear classifier. Returns raw logits."""
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.gnn = GatedGraphNeuralNetwork(
            hidden_size=hidden_size,
            num_edge_types=NUM_EDGE_TYPES,
            layer_timesteps=LAYER_TIMESTEPS,
            residual_connections=RESIDUAL,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, node_feats: torch.Tensor,
                adj_lists: list) -> torch.Tensor:
        """
        node_feats : [N, input_dim]
        adj_lists  : list of AdjacencyList (one per edge type, may be empty)
        Returns    : scalar logit
        """
        node_out = self.gnn(node_feats, adj_lists)        # [N, hidden_size]
        graph_emb = node_out.mean(dim=0, keepdim=True)    # [1, hidden_size]
        return self.classifier(graph_emb).squeeze()        # scalar

    def embed(self, node_feats: torch.Tensor,
              adj_lists: list) -> torch.Tensor:
        """Return graph-level embedding (no classifier)."""
        node_out = self.gnn(node_feats, adj_lists)
        return node_out.mean(dim=0)   # [hidden_size]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_graphs(json_path: Path, device: torch.device):
    """Load GGNN JSON and return list of (node_feats, adj_lists, target)."""
    with open(str(json_path)) as f:
        raw = json.load(f)

    graphs = []
    edge_vocab = {}   # edge_type_int → local_index (shared across call)

    for rec in raw:
        nf = torch.tensor(rec["node_features"], dtype=torch.float32)
        target = int(rec["targets"][0][0])

        # Group edges by type
        edges_by_type: dict = {}
        for s, etype, t in rec["graph"]:
            if etype not in edge_vocab:
                edge_vocab[etype] = len(edge_vocab)
            idx = edge_vocab[etype]
            if idx not in edges_by_type:
                edges_by_type[idx] = []
            edges_by_type[idx].append((s, t))

        n_nodes = len(rec["node_features"])
        adj_lists = []
        for eidx in range(NUM_EDGE_TYPES):
            edges = edges_by_type.get(eidx, [])
            adj_lists.append(AdjacencyList(
                node_num=n_nodes,
                adj_list=edges if edges else [(0, 0)],  # placeholder if empty
                device=device
            ))

        graphs.append((nf, adj_lists, target))

    return graphs, edge_vocab


def graphs_to_device(graphs, device):
    return [(nf.to(device), adjs, tgt) for nf, adjs, tgt in graphs]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def evaluate(model, graphs, device, threshold=0.5):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for nf, adjs, tgt in graphs:
            logit = model(nf.to(device), adjs).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            preds.append(1 if prob >= threshold else 0)
            targets.append(tgt)
    model.train()
    f1  = f1_score(targets, preds, zero_division=0) * 100
    acc = accuracy_score(targets, preds) * 100
    pr  = precision_score(targets, preds, zero_division=0) * 100
    rc  = recall_score(targets, preds, zero_division=0) * 100
    return acc, pr, rc, f1


def find_threshold(model, graphs, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for nf, adjs, tgt in graphs:
            logit = model(nf.to(device), adjs).item()
            scores.append(torch.sigmoid(torch.tensor(logit)).item())
            targets.append(tgt)
    model.train()
    best_f1, best_thr = -1, 0.5
    for thr in np.arange(0.05, 0.96, 0.05):
        p = [1 if s >= thr else 0 for s in scores]
        f1 = f1_score(targets, p, zero_division=0) * 100
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def train(model, train_graphs, valid_graphs, optimizer, scheduler, loss_fn, device,
          max_epochs=100, max_patience=20, log_every=5):
    best_f1 = -1
    best_state = None
    best_thr = 0.5
    patience = 0
    indices = list(range(len(train_graphs)))

    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0

        for idx in indices:
            nf, adjs, tgt = train_graphs[idx]
            nf = nf.to(device)
            target_t = torch.tensor([float(tgt)], device=device)
            optimizer.zero_grad()
            logit = model(nf, adjs)
            loss = loss_fn(logit.unsqueeze(0), target_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_graphs)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            thr, val_f1 = find_threshold(model, valid_graphs, device)
            acc, pr, rc, _ = evaluate(model, valid_graphs, device, thr)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                  f"val_F1={val_f1:.2f}%  Acc={acc:.2f}%  Pr={pr:.2f}%  Rc={rc:.2f}%  thr={thr:.2f}",
                  flush=True)

            scheduler.step(-val_f1)   # ReduceLROnPlateau minimises, so pass -F1

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_thr = thr
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
                print(f"  ✓ New best: F1={best_f1:.2f}%", flush=True)
            else:
                patience += 1
                print(f"  No improvement  patience={patience}/{max_patience}", flush=True)
                if patience >= max_patience:
                    print("  Early stopping.", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_thr, best_f1


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(model, graphs, device):
    model.eval()
    records = []
    with torch.no_grad():
        for nf, adjs, tgt in tqdm(graphs):
            emb = model.embed(nf.to(device), adjs).cpu().tolist()
            records.append({"graph_feature": emb, "target": tgt})
    return records


def save_after_ggnn(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(records, f)
    pos = sum(r["target"] for r in records)
    print(f"  Saved {len(records)} records  pos={pos}  neg={len(records)-pos}  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    print("Loading train graphs…", flush=True)
    train_graphs, edge_vocab = load_graphs(DEVIGN_INPUT / "train_GGNNinput.json", device)
    print(f"  {len(train_graphs)} train graphs  edge_types={len(edge_vocab)}")

    print("Loading valid graphs…", flush=True)
    valid_graphs, _ = load_graphs(DEVIGN_INPUT / "valid_GGNNinput.json", device)
    print(f"  {len(valid_graphs)} valid graphs")

    print("Loading test graphs…", flush=True)
    test_graphs, _ = load_graphs(DEVIGN_INPUT / "test_GGNNinput.json", device)
    print(f"  {len(test_graphs)} test graphs")

    # Infer input dim
    input_dim = train_graphs[0][0].shape[1]
    print(f"Input dim: {input_dim}", flush=True)

    # Class balance
    pos = sum(t for _, _, t in train_graphs)
    neg = len(train_graphs) - pos
    pos_weight = torch.tensor([neg / pos], device=device)
    print(f"pos={pos}  neg={neg}  pos_weight={pos_weight.item():.4f}", flush=True)

    # Model + loss
    model = REVEALVulnDetector(input_dim=input_dim).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7, verbose=False
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # Train
    print("\nTraining…", flush=True)
    model, best_thr, best_f1 = train(
        model, train_graphs, valid_graphs, optimizer, scheduler, loss_fn, device,
        max_epochs=args.max_epochs, max_patience=args.max_patience
    )

    # Test
    print("\nFinal test evaluation…", flush=True)
    acc, pr, rc, f1 = evaluate(model, test_graphs, device, best_thr)
    print(f"Test  Acc={acc:.2f}%  Pr={pr:.2f}%  Rc={rc:.2f}%  F1={f1:.2f}%  thr={best_thr:.2f}",
          flush=True)

    # Save model
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_OUT / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "best_threshold": best_thr,
                "best_val_f1": best_f1,
                "test_f1": f1,
                "input_dim": input_dim}, str(ckpt))
    print(f"Model saved: {ckpt}", flush=True)

    # Only proceed to extraction if model genuinely learned
    if f1 < 60.0:
        print(f"WARNING: Test F1={f1:.2f}% < 60% — model may be degenerate. "
              "Extracting embeddings anyway.", flush=True)

    # Extract embeddings for all splits
    print("\nExtracting embeddings…", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, graphs in [("train", train_graphs),
                          ("valid", valid_graphs),
                          ("test_originals", test_graphs)]:
        recs = extract_embeddings(model, graphs, device)
        fname = {"train": "train_GGNNinput_graph.json",
                 "valid": "valid_GGNNinput_graph.json",
                 "test_originals": "test_GGNNinput_graph.json"}[name]
        save_after_ggnn(recs, OUT_DIR / fname)

    # Obfuscated test sets
    for obf_name, obf_dir in OBF_DIRS.items():
        json_path = obf_dir / "test_GGNNinput.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found — skipping {obf_name}")
            continue
        obf_graphs, _ = load_graphs(json_path, device)
        recs = extract_embeddings(model, obf_graphs, device)
        fname = f"{obf_name}_test_GGNNinput_graph.json"
        save_after_ggnn(recs, OUT_DIR / fname)

    print(f"\nAll embeddings saved to {OUT_DIR}", flush=True)

    # Run robustness eval
    print("\nRunning REVEAL robustness eval…", flush=True)
    import subprocess
    reveal_rl = THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe/representation_learning"
    result = subprocess.run(
        [sys.executable, "-u", "reveal_robustness_eval.py",
         "--after_ggnn_dir", str(OUT_DIR),
         "--output_json", "robustness_results_reveal_ggnn.json"],
        cwd=str(reveal_rl),
        capture_output=False
    )
    print(f"Robustness eval exit code: {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
