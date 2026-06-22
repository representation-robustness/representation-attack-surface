#!/usr/bin/env python3
"""
Exp 4: Extract per-function ReGVD predictions for all 7 test conditions.
Uses the best.pt checkpoint from baselines/regvd/models/regvd_devign/.

Output: devign_full/attack/preds/regvd_preds.json
"""

import json, os, sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

THESIS     = Path(__file__).resolve().parents[1]
DEVIGN     = THESIS / "devign_full"
CKPT       = THESIS / "baselines" / "regvd" / "models" / "regvd_devign" / "best.pt"
REGVD_DIR  = THESIS / "baselines" / "regvd"
OUT_DIR    = DEVIGN / "attack" / "preds"
OUT_FILE   = OUT_DIR / "regvd_preds.json"
SPLIT_FILE = DEVIGN / "devign_full_split_801010.json"

sys.path.insert(0, str(REGVD_DIR))
from train_regvd import (
    ReGVDDataset, ReGVDModel, evaluate,
    CODEBERT_MODEL, HIDDEN_DIM, NUM_GNN_LAYERS, BATCH_SIZE,
)

CONDITIONS = {
    "clean":    DEVIGN / "originals_full_data_with_slices.json",
    "ren":      DEVIGN / "obf_identifier_full_data_with_slices.json",
    "dead":     DEVIGN / "obf_deadcode_full_data_with_slices.json",
    "cf":       DEVIGN / "obf_controlflow_full_data_with_slices.json",
    "ren_dead": DEVIGN / "obf_ren_dead_full_data_with_slices.json",
    "ren_cf":   DEVIGN / "obf_ren_cf_full_data_with_slices.json",
    "dead_cf":  DEVIGN / "obf_dead_cf_full_data_with_slices.json",
    "compound": DEVIGN / "obf_compound_full_data_with_slices.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

# Load split
split      = json.loads(SPLIT_FILE.read_text())
split      = split.get("splits", split)
test_files = split["test"]

# Load CodeBERT embedding weights (frozen lookup table, no full model needed)
from transformers import RobertaTokenizer, RobertaModel
print(f"Loading CodeBERT embedding weights from {CODEBERT_MODEL}...", flush=True)
tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
codebert_tmp = RobertaModel.from_pretrained(CODEBERT_MODEL)
embed_weight = codebert_tmp.embeddings.word_embeddings.weight.detach().cpu()
del codebert_tmp
print(f"  Vocab: {embed_weight.shape[0]:,}  dim: {embed_weight.shape[1]}", flush=True)

# Load clean data to get true labels
clean_data  = json.loads(Path(CONDITIONS["clean"]).read_text())
clean_idx   = {r["file_name"]: r for r in clean_data}
clean_recs  = [clean_idx[f] for f in test_files if f in clean_idx]
true_labels = [int(r["label"]) for r in clean_recs]
N           = len(true_labels)
print(f"Test set size: {N}", flush=True)

# Load model
state = torch.load(CKPT, map_location=DEVICE)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS).to(DEVICE)
model.load_state_dict(state, strict=False)
model.eval()
print(f"Loaded checkpoint: {CKPT}", flush=True)

def infer(records):
    ds     = ReGVDDataset(records, embed_weight, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    preds  = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)
            p      = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
    return preds

OUT_DIR.mkdir(parents=True, exist_ok=True)

seed_preds = {}
for cond, data_path in CONDITIONS.items():
    if not Path(data_path).exists():
        print(f"  {cond}: data file not found, skipping", flush=True)
        continue
    data = json.loads(Path(data_path).read_text())
    idx  = {r["file_name"]: r for r in data}
    recs = [idx[f] for f in test_files if f in idx]
    if not recs:
        print(f"  {cond}: 0 records, skipping", flush=True)
        continue
    n    = min(N, len(recs))
    preds = infer(recs[:n])
    if len(preds) < N:
        preds = preds + [0] * (N - len(preds))
    f1 = f1_score(true_labels[:n], preds[:n], zero_division=0) * 100
    print(f"  {cond}: F1={f1:.2f}%", flush=True)
    seed_preds[cond] = preds[:N]

out = {
    "model":       "regvd",
    "conditions":  list(CONDITIONS.keys()),
    "true_labels": true_labels,
    "seeds":       {"best": seed_preds},
}
OUT_FILE.write_text(json.dumps(out))
print(f"\nSaved → {OUT_FILE}", flush=True)
