#!/usr/bin/env python3
"""
Exp 12: VulGNN-WithId — causal ablation for identifier exposure.

Augments VulGNN with a function-level CodeBERT [CLS] embedding (768-dim)
concatenated with the GNN graph pooling output (128-dim) → 896-dim head.

This tests whether adding strong identifier representation to VulGNN makes it
vulnerable to identifier renaming — proving causation (not just correlation).

Pipeline:
  1. Recover GGNN record → filename mapping from parsed_cache + split file
  2. Pre-compute CodeBERT [CLS] embeddings for all matched functions
  3. Train VulGNN-WithId with GNN + CodeBERT for 5 seeds
  4. Evaluate on all 8 robustness conditions

Output: devign_full/vulgnn_withid_seedSEED_results.json
"""

import argparse, copy, csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GeneralConv, GraphNorm, global_mean_pool
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR   = Path(__file__).resolve().parent
THESIS_ROOT  = SCRIPT_DIR.parent
DEVIGN       = THESIS_ROOT / "devign_full"
DEVIGN_INPUT = DEVIGN / "devign_input"
PARSED_CACHE = DEVIGN_INPUT / "parsed_cache" / "originals"

CODEBERT_MODEL = "microsoft/codebert-base"
CODEBERT_CACHE = DEVIGN / "codebert_devign_embs"  # cached embeddings dir
CODEBERT_CACHE.mkdir(exist_ok=True)

# VulGNN hyperparams (same as baseline)
EDGE_REMAP     = {3: 0, 6: 1, 7: 2, 9: 3, 10: 4}
NUM_EDGE_TYPES = 5
EDGE_EMB_DIM   = 4
IN_DIM         = 169
HIDDEN         = 128
NUM_LAYERS     = 6
DROPOUT        = 0.08
BATCH_SIZE     = 256
LR             = 1e-4
FOCAL_GAMMA    = 2.0
MAX_EPOCHS     = 60
PATIENCE       = 15
SEEDS          = [42, 1337, 7, 100, 999]

CODEBERT_DIM   = 768
COMBINED_DIM   = HIDDEN + CODEBERT_DIM   # 896


# ---------------------------------------------------------------------------
# Step 1: Recover GGNN record → filename mapping
# ---------------------------------------------------------------------------

def recover_ggnn_filenames(split_names: list, parsed_root: Path) -> list:
    """
    Simulate build_graph_json filtering to get the ordered list of filenames
    that end up as GGNN records. Returns a list of filenames in the same order
    as the GGNN JSON records.
    """
    included = []
    for fname in split_names:
        node_csv = parsed_root / fname / "nodes.csv"
        edge_csv = parsed_root / fname / "edges.csv"
        if not node_csv.exists() or not edge_csv.exists():
            continue
        count = 0
        with open(node_csv) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row['type'] == 'File':
                    continue
                cfg = row['isCFGNode'].strip()
                if cfg == '' or cfg == 'False':
                    continue
                count += 1
        if count == 0 or count >= 500:
            continue
        has_edges = False
        with open(edge_csv) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row['type'] != 'IS_FILE_OF':
                    has_edges = True
                    break
        if has_edges:
            included.append(fname)
    return included


# ---------------------------------------------------------------------------
# Step 2: CodeBERT embedding computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_codebert_embs(texts: list, device, batch_size=32, max_len=512) -> np.ndarray:
    """Compute CodeBERT [CLS] embeddings for a list of function texts."""
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model     = AutoModel.from_pretrained(CODEBERT_MODEL).to(device).eval()
    all_embs  = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt",
                        max_length=max_len, truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls)
        if (i // batch_size) % 50 == 0:
            print(f"  CodeBERT embs: {i}/{len(texts)}", flush=True)
    return np.vstack(all_embs)


def get_or_compute_embs(split_key: str, filenames: list, fname2code: dict,
                        device) -> np.ndarray:
    cache_file = CODEBERT_CACHE / f"{split_key}_embs.npy"
    fnames_file = CODEBERT_CACHE / f"{split_key}_fnames.json"
    if cache_file.exists() and fnames_file.exists():
        cached_fnames = json.loads(fnames_file.read_text())
        if cached_fnames == filenames:
            print(f"  Loading cached CodeBERT embs for {split_key}…", flush=True)
            return np.load(str(cache_file))
    print(f"  Computing CodeBERT embs for {split_key} ({len(filenames)} functions)…",
          flush=True)
    texts = [fname2code[f] for f in filenames]
    embs  = compute_codebert_embs(texts, device)
    np.save(str(cache_file), embs)
    fnames_file.write_text(json.dumps(filenames))
    print(f"  Saved to {cache_file}", flush=True)
    return embs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs(json_path: Path, embs: np.ndarray = None) -> list:
    with open(json_path) as f:
        records = json.load(f)
    graphs = []
    for i, rec in enumerate(records):
        nf     = torch.tensor(rec["node_features"], dtype=torch.float32)
        target = int(rec["targets"][0][0])
        raw    = rec.get("graph", [])
        if raw:
            srcs   = [e[0] for e in raw]
            dsts   = [e[2] for e in raw]
            etypes = [EDGE_REMAP.get(e[1], 0) for e in raw]
            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            edge_type  = torch.tensor(etypes, dtype=torch.long)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_type  = torch.tensor([0], dtype=torch.long)
        y = torch.tensor([float(target)], dtype=torch.float32)
        d = Data(x=nf, edge_index=edge_index, edge_type=edge_type, y=y)
        if embs is not None:
            # Shape [1, 768] so DataLoader stacks to [B, 768] (not [B*768])
            d.cb_emb = torch.tensor(embs[i], dtype=torch.float32).unsqueeze(0)
        graphs.append(d)
    return graphs


def balanced_loader(graphs, batch_size):
    labels = [int(g.y.item()) for g in graphs]
    pos = sum(labels); neg = len(labels) - pos
    w   = [1.0 / neg if l == 0 else 1.0 / pos for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler,
                      follow_batch=[])


def plain_loader(graphs, batch_size):
    return DataLoader(graphs, batch_size=batch_size, shuffle=False,
                      follow_batch=[])


# ---------------------------------------------------------------------------
# Model: VulGNN + CodeBERT head
# ---------------------------------------------------------------------------

class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.conv = GeneralConv(in_channels, out_channels,
                                in_edge_channels=edge_dim, aggr='mean',
                                attention=True, attention_type='dot_product')
        self.act  = nn.PReLU()
        self.norm = GraphNorm(out_channels)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.conv(x, edge_index, edge_attr)
        return self.drop(self.norm(self.act(h), batch))


class VulGNNWithId(nn.Module):
    """VulGNN GNN backbone + CodeBERT function-level embedding in classification head."""

    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, num_layers=NUM_LAYERS,
                 num_edge_types=NUM_EDGE_TYPES, edge_emb_dim=EDGE_EMB_DIM,
                 codebert_dim=CODEBERT_DIM):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_emb   = nn.Embedding(num_edge_types, edge_emb_dim)
        self.blocks     = nn.ModuleList([
            ConvGroup(hidden, hidden, edge_emb_dim) for _ in range(num_layers)
        ])
        # Head takes GNN pooling (128) + CodeBERT CLS (768) = 896
        self.head = nn.Sequential(
            nn.Linear(COMBINED_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x, edge_index, edge_type, batch, cb_emb):
        h  = self.input_proj(x)
        ea = self.edge_emb(edge_type)
        for block in self.blocks:
            h = block(h, edge_index, ea, batch)
        g   = global_mean_pool(h, batch)          # (B, 128)
        # cb_emb is batched as [B, 768] from [1, 768] per-graph tensors
        combined = torch.cat([g, cb_emb], dim=1)  # (B, 896)
        return self.head(combined)


# ---------------------------------------------------------------------------
# Loss & eval
# ---------------------------------------------------------------------------

def focal_loss(logits, labels_float, gamma=FOCAL_GAMMA):
    binary_logit = logits[:, 1] - logits[:, 0]
    bce = F.binary_cross_entropy_with_logits(binary_logit, labels_float, reduction='none')
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.edge_type, b.batch, b.cb_emb)
        logits_all.append(out.cpu())
        labels_all.append(b.y.cpu())
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy().astype(int).flatten()
    preds  = logits.argmax(axis=1)
    return preds, labels


def eval_metrics(preds, labels):
    return {
        "acc": accuracy_score(labels, preds) * 100,
        "pr":  precision_score(labels, preds, zero_division=0) * 100,
        "rc":  recall_score(labels, preds, zero_division=0) * 100,
        "f1":  f1_score(labels, preds, zero_division=0) * 100,
        "n":   int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_graphs, valid_graphs, device):
    optimizer   = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    no_improve  = 0
    val_loader  = plain_loader(valid_graphs, BATCH_SIZE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_b = 0.0, 0
        for b in balanced_loader(train_graphs, BATCH_SIZE):
            b = b.to(device)
            out  = model(b.x, b.edge_index, b.edge_type, b.batch, b.cb_emb)
            loss = focal_loss(out, b.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_b += 1

        val_preds, val_labels = predict(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0) * 100
        val_rc = recall_score(val_labels, val_preds, zero_division=0) * 100
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  loss={total_loss/max(n_b,1):.4f}  "
              f"val_F1={val_f1:.2f}%  val_Rc={val_rc:.2f}%", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
            torch.save(best_state, DEVIGN / f"vulgnn_withid_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VulGNN-WithId seed={args.seed}] Device: {device}", flush=True)

    # ── Step 1: Recover GGNN record → filename mapping ──────────────────────
    print("Recovering GGNN record → filename mapping…", flush=True)
    split = json.loads((DEVIGN / 'devign_full_split_801010.json').read_text())['splits']
    full_data = json.loads((DEVIGN / 'originals_full_data_with_slices.json').read_text())
    fname2code = {r['file_name']: r['code'] for r in full_data}

    train_fnames = recover_ggnn_filenames(split['train'], PARSED_CACHE)
    valid_fnames = recover_ggnn_filenames(split['valid'], PARSED_CACHE)
    test_fnames  = recover_ggnn_filenames(split['test'],  PARSED_CACHE)
    print(f"  Mapped: train={len(train_fnames)} valid={len(valid_fnames)} "
          f"test={len(test_fnames)}", flush=True)

    # Verify counts match GGNN JSON
    with open(DEVIGN_INPUT / "originals_train" / "train_GGNNinput.json") as f:
        n_train = len(json.load(f))
    with open(DEVIGN_INPUT / "originals_train" / "valid_GGNNinput.json") as f:
        n_valid = len(json.load(f))
    with open(DEVIGN_INPUT / "originals_train" / "test_GGNNinput.json") as f:
        n_test  = len(json.load(f))
    print(f"  GGNN JSON counts: train={n_train} valid={n_valid} test={n_test}", flush=True)
    if len(train_fnames) != n_train or len(valid_fnames) != n_valid or len(test_fnames) != n_test:
        print("WARNING: Mapping size mismatch! Proceeding anyway…", flush=True)
        # Trim to match GGNN JSON size if necessary
        train_fnames = train_fnames[:n_train]
        valid_fnames = valid_fnames[:n_valid]
        test_fnames  = test_fnames[:n_test]

    # ── Step 2: Compute CodeBERT embeddings ──────────────────────────────────
    print("\nComputing/loading CodeBERT embeddings…", flush=True)
    train_embs = get_or_compute_embs("train", train_fnames, fname2code, device)
    valid_embs = get_or_compute_embs("valid", valid_fnames, fname2code, device)
    test_embs  = get_or_compute_embs("test",  test_fnames,  fname2code, device)

    # For obf test sets: compute CodeBERT embeddings on the obfuscated function texts
    obf_source_files = {
        "ren":      DEVIGN / "obf_identifier_full_data_with_slices.json",
        "dead":     DEVIGN / "obf_deadcode_full_data_with_slices.json",
        "cf":       DEVIGN / "obf_controlflow_full_data_with_slices.json",
        "ren_dead": DEVIGN / "obf_ren_dead_full_data_with_slices.json",
        "ren_cf":   DEVIGN / "obf_ren_cf_full_data_with_slices.json",
        "dead_cf":  DEVIGN / "obf_dead_cf_full_data_with_slices.json",
        "compound": DEVIGN / "obf_compound_full_data_with_slices.json",
    }
    obf_embs = {}
    for cond, obf_file in obf_source_files.items():
        cond_file   = CODEBERT_CACHE / f"test_{cond}_embs.npy"
        fnames_file = CODEBERT_CACHE / f"test_{cond}_fnames.json"
        if cond_file.exists() and fnames_file.exists():
            cached = json.loads(fnames_file.read_text())
            if cached == test_fnames:
                print(f"  Loading cached CodeBERT embs for test {cond}…", flush=True)
                obf_embs[cond] = np.load(str(cond_file))
                continue
        print(f"  Computing CodeBERT embs for test {cond}…", flush=True)
        obf_data = json.loads(obf_file.read_text())
        obf_fname2code = {r['file_name']: r['code'] for r in obf_data}
        texts = [obf_fname2code.get(f, fname2code.get(f, "")) for f in test_fnames]
        embs  = compute_codebert_embs(texts, device)
        np.save(str(cond_file), embs)
        fnames_file.write_text(json.dumps(test_fnames))
        obf_embs[cond] = embs

    # ── Step 3: Load graphs with CodeBERT embeddings ─────────────────────────
    print("\nLoading graphs…", flush=True)
    train_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "train_GGNNinput.json",
                               train_embs)
    valid_graphs = load_graphs(DEVIGN_INPUT / "originals_train" / "valid_GGNNinput.json",
                               valid_embs)
    print(f"  train={len(train_graphs)}  valid={len(valid_graphs)}", flush=True)

    test_graph_data = {
        "clean":    (DEVIGN_INPUT / "originals_train"    / "test_GGNNinput.json",      test_embs),
        "ren":      (DEVIGN_INPUT / "obf_identifier_test" / "test_GGNNinput.json",     obf_embs["ren"]),
        "dead":     (DEVIGN_INPUT / "obf_deadcode_test"   / "test_GGNNinput.json",     obf_embs["dead"]),
        "cf":       (DEVIGN_INPUT / "obf_controlflow_test"/ "test_GGNNinput.json",     obf_embs["cf"]),
        "ren_dead": (DEVIGN_INPUT / "pairwise_test"       / "obf_ren_dead_test_GGNNinput.json", obf_embs["ren_dead"]),
        "ren_cf":   (DEVIGN_INPUT / "pairwise_test"       / "obf_ren_cf_test_GGNNinput.json",   obf_embs["ren_cf"]),
        "dead_cf":  (DEVIGN_INPUT / "pairwise_test"       / "obf_dead_cf_test_GGNNinput.json",  obf_embs["dead_cf"]),
        "compound": (DEVIGN_INPUT / "obf_compound_test"   / "test_GGNNinput.json",     obf_embs["compound"]),
    }

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print(f"\nTraining VulGNN-WithId seed={args.seed}…", flush=True)
    model = VulGNNWithId().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}", flush=True)
    model = train(model, train_graphs, valid_graphs, device)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    results = {}
    clean_f1 = None

    for cond, (path, embs) in test_graph_data.items():
        graphs = load_graphs(path, embs)
        loader = plain_loader(graphs, BATCH_SIZE)
        preds, labels = predict(model, loader, device)
        m = eval_metrics(preds, labels)
        results[cond] = m
        if clean_f1 is None:
            clean_f1 = m["f1"]
        delta = m["f1"] - clean_f1
        print(f"  {cond:8s}  F1={m['f1']:.2f}%  ΔF1={delta:+.2f}pp", flush=True)

    results["_meta"] = {
        "model":         "vulgnn_withid",
        "seed":          args.seed,
        "codebert_dim":  CODEBERT_DIM,
        "gnn_hidden":    HIDDEN,
        "combined_dim":  COMBINED_DIM,
    }

    out = DEVIGN / f"vulgnn_withid_seed{args.seed}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}", flush=True)

    print("\n=== VulGNN-WithId Delta-F1 Summary ===", flush=True)
    for cond in ["ren", "dead", "cf", "compound"]:
        delta = results[cond]["f1"] - clean_f1
        print(f"  Δ{cond:8s} = {delta:+.2f}pp", flush=True)


if __name__ == "__main__":
    main()
