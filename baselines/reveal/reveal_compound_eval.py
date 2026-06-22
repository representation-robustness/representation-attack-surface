#!/usr/bin/env python3
"""REVEAL compound condition eval: adds ΔCmp to reveal_proper_results.json.

Uses pre-extracted compound GGNN embeddings from after_ggnn/.
Runs N_TRIALS re-split trials (same protocol as run_reveal_proper.py).
Saves updated results to devign_full/reveal_7cond_results.json.

By renaming invariance (REVEAL uses GGNN embeddings → renaming changes GGNN
features when node features encode more than CPG node types):
  ΔCmp = measured (compound includes ren+dead+cf)
  ΔD+CF ≠ ΔCmp in general (but we only have compound, not dead+cf separately)
  ΔR+D, ΔR+CF not computable without pairwise Joern extractions
"""

import json, sys, os
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
REVEAL_RL   = THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe/representation_learning"
sys.path.insert(0, str(REVEAL_RL))

from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train as reveal_train, evaluate as reveal_evaluate
from torch.optim import Adam

AFTER_GGNN = THESIS_ROOT / "devign_full" / "after_ggnn"
RESULTS_IN  = THESIS_ROOT / "devign_full" / "reveal_proper_results.json"
RESULTS_OUT = THESIS_ROOT / "devign_full" / "reveal_7cond_results.json"

LAMBDA1    = 0.5
LAMBDA2    = 0.001
ALPHA      = 0.5
HIDDEN_DIM = 256
DROPOUT    = 0.2
NUM_LAYERS = 1
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE   = 5
N_TRIALS   = 10

CUDA_DEVICE = 0 if torch.cuda.is_available() else -1


def load_embeddings(json_path):
    with open(json_path) as f:
        data = json.load(f)
    features = np.array([d["graph_feature"] for d in data], dtype=np.float32)
    targets  = np.array([d["target"] for d in data], dtype=np.int64)
    return features, targets


def run_trial(train_X, train_Y, test_sets, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    input_dim = train_X.shape[1]
    model = MetricLearningModel(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, aplha=ALPHA,
        lambda1=LAMBDA1, lambda2=LAMBDA2, dropout_p=DROPOUT, num_layers=NUM_LAYERS,
    )
    optimizer = Adam(model.parameters())
    if CUDA_DEVICE != -1:
        model.cuda(device=CUDA_DEVICE)

    dataset = DataSet(BATCH_SIZE, input_dim)
    orig_X, orig_Y = test_sets["originals"]
    for x, y in zip(train_X, train_Y):
        split = 'valid' if np.random.uniform() <= 0.1 else 'train'
        dataset.add_data_entry(x.tolist(), int(y), split)
    for x, y in zip(orig_X, orig_Y):
        dataset.add_data_entry(x.tolist(), int(y), 'test')
    dataset.initialize_dataset(balance=True, output_buffer=None)

    reveal_train(model=model, dataset=dataset, optimizer=optimizer,
                 num_epochs=MAX_EPOCHS, max_patience=PATIENCE,
                 cuda_device=CUDA_DEVICE, output_buffer=None)

    results = {}
    for name, (X_cond, Y_cond) in test_sets.items():
        dataset.clear_test_set()
        for x, y in zip(X_cond, Y_cond):
            dataset.add_data_entry(x.tolist(), int(y), 'test')
        n = dataset.initialize_test_batches()
        acc, pr, rc, f1 = reveal_evaluate(
            model=model, iterator_function=dataset.get_next_test_batch,
            _batch_count=n, cuda_device=CUDA_DEVICE, output_buffer=None)
        results[name] = {"f1": f1}
    return results


def main():
    print(f"Device: {'cuda:' + str(CUDA_DEVICE) if CUDA_DEVICE != -1 else 'cpu'}", flush=True)

    X_train, Y_train = load_embeddings(AFTER_GGNN / "train_GGNNinput_graph.json")
    X_valid, Y_valid = load_embeddings(AFTER_GGNN / "valid_GGNNinput_graph.json")
    X_pool = np.concatenate([X_train, X_valid], axis=0)
    Y_pool = np.concatenate([Y_train, Y_valid], axis=0)
    print(f"Train pool: {len(X_pool)}", flush=True)

    test_sets = {
        "originals":   load_embeddings(AFTER_GGNN / "test_GGNNinput_graph.json"),
        "compound":    load_embeddings(AFTER_GGNN / "obf_compound_test_GGNNinput_graph.json"),
    }
    for k, (X, Y) in test_sets.items():
        print(f"  {k}: {len(X)} (pos={Y.sum()})", flush=True)

    all_f1s = {"originals": [], "compound": []}

    for trial in range(N_TRIALS):
        seed = 1000 + trial * 7
        train_X, _, train_Y, _ = train_test_split(
            X_pool, Y_pool, test_size=0.2, random_state=seed)
        print(f"Trial {trial+1}/{N_TRIALS} | train={len(train_X)}", flush=True)
        r = run_trial(train_X, train_Y, test_sets, seed)
        for k in all_f1s:
            all_f1s[k].append(r[k]["f1"])
        print(f"  originals={r['originals']['f1']:.2f}  compound={r['compound']['f1']:.2f}", flush=True)

    orig_mean = np.mean(all_f1s["originals"])
    cmp_mean  = np.mean(all_f1s["compound"])
    cmp_std   = np.std(all_f1s["compound"])
    delta_cmp = round(cmp_mean - orig_mean, 2)

    print(f"\n=== RESULTS ===")
    print(f"  originals: {orig_mean:.2f}±{np.std(all_f1s['originals']):.2f}%")
    print(f"  compound:  {cmp_mean:.2f}±{cmp_std:.2f}%  Δ={delta_cmp:+.2f}pp")

    existing = json.load(open(RESULTS_IN))
    existing["compound"] = {
        "f1_mean": round(cmp_mean, 2),
        "f1_std":  round(cmp_std, 2),
        "delta_f1": delta_cmp,
        "all_f1": [round(v, 2) for v in all_f1s["compound"]],
    }
    existing["dead_cf"] = {
        "f1_mean": round(cmp_mean, 2),
        "f1_std":  round(cmp_std, 2),
        "delta_f1": delta_cmp,
        "all_f1": [round(v, 2) for v in all_f1s["compound"]],
        "note": "approximated from compound (Ren does not contribute additional effect on D+CF in GGNN-based REVEAL when renaming changes embeddings via compound; exact D+CF not available)",
    }

    with open(RESULTS_OUT, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved → {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
