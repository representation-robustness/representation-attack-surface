#!/usr/bin/env python3
"""
ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection
Nguyen et al., ICSE '22 Companion

Key design (no Joern, no CPG):
  - View raw source code as flat BPE token sequence
  - Build unique-token-focused co-occurrence graph:
      nodes = unique BPE token IDs
      edges = co-occurrence within a sliding window of size WINDOW_SIZE
  - Node features = CodeBERT static token embeddings (768-dim, frozen)
  - GCN with residual connections + LayerNorm
  - Soft-attention readout: e_v = sigmoid(gate(h)) * tanh(trans(h))
  - Graph embedding = CONCAT(sum_pool(e_v), max_pool(e_v))
  - Classifier: single linear layer

Why this should escape degenerate collapse:
  - Token co-occurrence graph carries discriminative signal (unlike CPG
    topology which has F1=48% standalone on this dataset)
  - 768-dim CodeBERT embeddings vs 100-dim Word2Vec — richer node features
  - No dependency on Joern, DGL, or any version-fragile infrastructure

Literature: ReGVD (GCN+UniT+G-CB) achieves 63.69% accuracy on CodeXGLUE
(same underlying FFmpeg/QEMU dataset, different split from ours).
"""

import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
DEVIGN_ROOT = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_ROOT / "devign_full_split_801010.json"
MODEL_DIR   = SCRIPT_DIR / "models" / "regvd_devign"

CODEBERT_MODEL  = "microsoft/codebert-base"
MAX_TOKENS      = 512      # BPE tokens per function (truncate)
WINDOW_SIZE     = 3        # co-occurrence sliding window
HIDDEN_DIM      = 256      # GNN hidden dimension
NUM_GNN_LAYERS  = 2        # number of GCN layers
BATCH_SIZE      = 128
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
NUM_EPOCHS      = 100
PATIENCE        = 10
FOCAL_GAMMA     = 2.0

DATA_FILES = {
    "originals":       DEVIGN_ROOT / "originals_full_data_with_slices.json",
    "obf_identifier":  DEVIGN_ROOT / "obf_identifier_full_data_with_slices.json",
    "obf_deadcode":    DEVIGN_ROOT / "obf_deadcode_full_data_with_slices.json",
    "obf_controlflow": DEVIGN_ROOT / "obf_controlflow_full_data_with_slices.json",
}


# ---------------------------------------------------------------------------
# Graph dataset
# ---------------------------------------------------------------------------

class ReGVDDataset(TorchDataset):
    """
    Unique-token-focused co-occurrence graph dataset.

    Stores graph topology as (unique_token_ids, edge_index, label).
    Node feature vectors x are computed lazily from embed_weight at
    __getitem__ time — only the topology is stored, saving ~10x RAM
    vs pre-computing full float32 feature matrices.
    """

    def __init__(self, records, embed_weight, tokenizer,
                 window_size=WINDOW_SIZE, max_tokens=MAX_TOKENS):
        self.embed_weight = embed_weight   # shape (vocab_size, 768), CPU
        self._graphs = []

        for i, rec in enumerate(records):
            if i > 0 and i % 5000 == 0:
                print(f"    {i}/{len(records)} graphs built...", flush=True)

            code  = rec["code"]
            label = int(rec["label"])

            # BPE tokenize (no special tokens — we want raw code tokens)
            enc = tokenizer(code, max_length=max_tokens, truncation=True,
                            add_special_tokens=False)
            token_ids = enc["input_ids"] or [tokenizer.unk_token_id]

            # Unique-token-focused construction: deduplicate, preserve order
            unique_ids   = list(dict.fromkeys(token_ids))
            tok_to_node  = {t: i for i, t in enumerate(unique_ids)}

            # Co-occurrence edges within sliding window (no self-loops)
            edge_set = set()
            for pos in range(len(token_ids)):
                for q in range(pos + 1, min(pos + window_size, len(token_ids))):
                    u = tok_to_node[token_ids[pos]]
                    v = tok_to_node[token_ids[q]]
                    if u != v:
                        edge_set.add((u, v))
                        edge_set.add((v, u))

            if edge_set:
                ei = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)

            self._graphs.append((
                torch.tensor(unique_ids, dtype=torch.long),
                ei,
                torch.tensor([label], dtype=torch.long),
            ))

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        unique_ids, edge_index, y = self._graphs[idx]
        # Embedding lookup: creates a new (num_unique, 768) tensor per call
        x = self.embed_weight[unique_ids]
        return Data(x=x, edge_index=edge_index, y=y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ReGVDModel(nn.Module):
    """
    GCN with residual connections + soft-attention sum/max readout.
    Follows the architecture in Nguyen et al. (2022).
    """

    def __init__(self, in_dim=768, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_GNN_LAYERS):
        super().__init__()

        # Project CodeBERT 768-dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )

        # GCN layers (paper omits self-loops)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Soft-attention readout:
        #   e_v = sigmoid(gate(h)) ⊙ tanh(trans(h))
        self.gate_lin  = nn.Linear(hidden_dim, 1)
        self.trans_lin = nn.Linear(hidden_dim, hidden_dim)

        # Classifier on CONCAT(sum_pool, max_pool)  →  2 logits
        self.clf = nn.Linear(hidden_dim * 2, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(conv(h, edge_index))
            h     = norm(h + h_new)          # residual + LayerNorm

        # Soft-attention readout (as in original ReGVD)
        gate = torch.sigmoid(self.gate_lin(h))
        feat = torch.tanh(self.trans_lin(h))
        e_v  = gate * feat

        # Sum + Max pooling, concatenated
        eg = torch.cat([global_add_pool(e_v, batch),
                        global_max_pool(e_v, batch)], dim=-1)

        return self.clf(eg)


# ---------------------------------------------------------------------------
# Loss, evaluation, training
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data   = data.to(device)
        preds  = model(data).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    pr  = precision_score(all_labels, all_preds, zero_division=0) * 100
    rc  = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1  = f1_score(all_labels, all_preds, zero_division=0) * 100
    return acc, pr, rc, f1


def train_model(model, train_loader, valid_loader, device):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    no_improve  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_steps    = 0
        for data in train_loader:
            data = data.to(device)
            loss = focal_loss(model(data), data.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_steps    += 1

        va, vp, vr, vf = evaluate(model, valid_loader, device)
        deg = "  [DEGEN]" if vr > 93.0 else ""
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  loss={total_loss/n_steps:.4f}  "
              f"val_F1={vf:.2f}%  val_Pr={vp:.2f}%  val_Rc={vr:.2f}%{deg}",
              flush=True)

        if vf > best_val_f1 + 0.1:
            best_val_f1 = vf
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
            print(f"  -> Best val F1={vf:.2f}%  Pr={vp:.2f}%  Rc={vr:.2f}%",
                  flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}  "
              f"{free/1e9:.1f}/{total/1e9:.1f} GB free", flush=True)

    # ---- Load split and data -----------------------------------------------
    print("\nLoading data...", flush=True)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_files = set(split["splits"]["train"])
    valid_files = set(split["splits"]["valid"])
    test_files  = set(split["splits"]["test"])

    with open(DATA_FILES["originals"]) as f:
        orig = json.load(f)
    idx = {d["file_name"]: d for d in orig}

    train_recs = [idx[f] for f in train_files if f in idx]
    valid_recs = [idx[f] for f in valid_files if f in idx]
    test_recs  = [idx[f] for f in test_files  if f in idx]

    pos = sum(int(r["label"]) for r in train_recs)
    neg = len(train_recs) - pos
    print(f"  train={len(train_recs)} (pos={pos}, neg={neg})  "
          f"valid={len(valid_recs)}  test={len(test_recs)}", flush=True)

    # ---- CodeBERT static embeddings ----------------------------------------
    print(f"\nLoading CodeBERT token embeddings from {CODEBERT_MODEL}...",
          flush=True)
    tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
    del codebert   # free the full transformer (we only need the embedding table)
    print(f"  Vocab: {embed_weight.shape[0]:,}  dim: {embed_weight.shape[1]}",
          flush=True)

    # ---- Build graph datasets ----------------------------------------------
    print("\nBuilding training graphs...", flush=True)
    train_ds = ReGVDDataset(train_recs, embed_weight, tokenizer)
    print("Building validation graphs...", flush=True)
    valid_ds = ReGVDDataset(valid_recs, embed_weight, tokenizer)
    print("Building test graphs...", flush=True)
    test_ds  = ReGVDDataset(test_recs,  embed_weight, tokenizer)
    print("Dataset construction complete.", flush=True)

    # Balanced 50/50 sampler for training
    labels   = [int(r["label"]) for r in train_recs]
    weights  = [1.0/neg if l == 0 else 1.0/pos for l in labels]
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights),
                                     replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0)

    # ---- Model -------------------------------------------------------------
    model    = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM,
                          num_layers=NUM_GNN_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nReGVD: {n_params:,} parameters  "
          f"(GCN layers={NUM_GNN_LAYERS}, hidden={HIDDEN_DIM}, "
          f"window={WINDOW_SIZE})", flush=True)
    print(f"Training: {NUM_EPOCHS} epochs max  batch={BATCH_SIZE}  "
          f"lr={LR}  patience={PATIENCE}", flush=True)

    # ---- Train -------------------------------------------------------------
    model, best_val_f1 = train_model(model, train_loader, valid_loader, device)

    # ---- Test --------------------------------------------------------------
    ta, tp, tr, tf = evaluate(model, test_loader, device)
    print(f"\nOriginal test:  Acc={ta:.2f}%  Pr={tp:.2f}%  "
          f"Rc={tr:.2f}%  F1={tf:.2f}%", flush=True)

    degenerate = tr > 93.0 or tf <= 63.0
    print(f"{'DEGENERATE' if degenerate else 'NON-DEGENERATE'}  "
          f"(Pr={tp:.1f}%, Rc={tr:.1f}%)", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "best.pt")
    print(f"Model saved → {MODEL_DIR}", flush=True)

    if degenerate:
        print("Model degenerate — skipping robustness eval.", flush=True)
        return

    # ---- Robustness evaluation ---------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("ROBUSTNESS EVALUATION", flush=True)
    print("=" * 60, flush=True)

    results = {"original": {"acc": ta, "pr": tp, "rc": tr, "f1": tf}}

    for obf_name, obf_file in [
        ("obf_identifier",  DATA_FILES["obf_identifier"]),
        ("obf_deadcode",    DATA_FILES["obf_deadcode"]),
        ("obf_controlflow", DATA_FILES["obf_controlflow"]),
    ]:
        with open(obf_file) as f:
            obf_data = json.load(f)
        obf_idx   = {d["file_name"]: d for d in obf_data}
        obf_recs  = [obf_idx[f] for f in test_files if f in obf_idx]
        obf_ds    = ReGVDDataset(obf_recs, embed_weight, tokenizer)
        obf_loader = DataLoader(obf_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                num_workers=0)
        acc, pr, rc, f1 = evaluate(model, obf_loader, device)
        delta = f1 - tf
        results[obf_name] = {"acc": acc, "pr": pr, "rc": rc,
                              "f1": f1, "delta_f1": delta}
        print(f"  {obf_name:20s}  Acc={acc:.2f}%  Pr={pr:.2f}%  "
              f"Rc={rc:.2f}%  F1={f1:.2f}%  ΔF1={delta:+.2f}%", flush=True)

    print(f"\nRobustness summary (ΔF1 from baseline {tf:.2f}%):", flush=True)
    for k, v in results.items():
        if k != "original":
            print(f"  {k:20s}  ΔF1={v['delta_f1']:+.2f}%", flush=True)

    # ---- Save results ------------------------------------------------------
    out = {
        "model":        "ReGVD",
        "gnn_type":     "GCN",
        "hidden_dim":   HIDDEN_DIM,
        "window_size":  WINDOW_SIZE,
        "best_val_f1":  best_val_f1,
        "test_original": results["original"],
        "robustness":   {k: v for k, v in results.items() if k != "original"},
    }
    results_path = SCRIPT_DIR / "regvd_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {results_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
