#!/usr/bin/env python3
"""
Robustness evaluation for saved ReGVD model.
Loads the saved checkpoint and evaluates on all 4 test conditions.
"""

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from train_regvd import (
    ReGVDDataset, ReGVDModel,
    HIDDEN_DIM, NUM_GNN_LAYERS, WINDOW_SIZE, BATCH_SIZE,
    SCRIPT_DIR, THESIS_ROOT, DEVIGN_ROOT, SPLIT_FILE,
    MODEL_DIR, CODEBERT_MODEL, DATA_FILES
)

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_files = set(split["splits"]["test"])

    with open(DATA_FILES["originals"]) as f:
        orig = json.load(f)
    idx = {d["file_name"]: d for d in orig}
    test_recs = [idx[f] for f in test_files if f in idx]

    # CodeBERT embeddings
    print("Loading CodeBERT embeddings...", flush=True)
    tokenizer    = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert     = RobertaModel.from_pretrained(CODEBERT_MODEL)
    embed_weight = codebert.embeddings.word_embeddings.weight.detach().cpu()
    del codebert

    # Load model
    model = ReGVDModel(in_dim=768, hidden_dim=HIDDEN_DIM,
                       num_layers=NUM_GNN_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    model.eval()
    print("Model loaded.", flush=True)

    # Build original test loader
    print("Building test datasets...", flush=True)
    test_ds = ReGVDDataset(test_recs, embed_weight, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=0)

    ta, tp, tr, tf = evaluate(model, test_loader, device)
    print(f"\nOriginal test:  Acc={ta:.2f}%  Pr={tp:.2f}%  "
          f"Rc={tr:.2f}%  F1={tf:.2f}%", flush=True)

    # Robustness
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
        obf_loader = DataLoader(obf_ds, batch_size=BATCH_SIZE * 2,
                                shuffle=False, num_workers=0)
        acc, pr, rc, f1 = evaluate(model, obf_loader, device)
        delta = f1 - tf
        results[obf_name] = {"acc": acc, "pr": pr, "rc": rc,
                              "f1": f1, "delta_f1": delta}
        print(f"  {obf_name:20s}  Acc={acc:.2f}%  Pr={pr:.2f}%  "
              f"Rc={rc:.2f}%  F1={f1:.2f}%  ΔF1={delta:+.2f}%", flush=True)

    out = {
        "model": "ReGVD",
        "gnn_type": "GCN",
        "hidden_dim": HIDDEN_DIM,
        "window_size": WINDOW_SIZE,
        "test_original": results["original"],
        "robustness": {k: v for k, v in results.items() if k != "original"},
    }
    results_path = SCRIPT_DIR / "regvd_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {results_path}", flush=True)


if __name__ == "__main__":
    main()
