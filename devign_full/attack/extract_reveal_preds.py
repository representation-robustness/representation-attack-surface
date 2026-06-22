#!/usr/bin/env python3
"""
Extract per-function predictions for REVEAL on 5 available conditions
(clean, ren, dead, cf, compound). Uses pre-extracted GGNN embeddings.

Runs N_TRIALS metric-learning trials (same protocol as run_reveal_proper.py).
Saves preds/reveal_preds.json.
"""

import json, sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

THESIS_ROOT = Path(__file__).resolve().parents[2]
REVEAL_RL   = THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe/representation_learning"
sys.path.insert(0, str(REVEAL_RL))

from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train as reveal_train, predict as reveal_predict
from torch.optim import Adam

AFTER_GGNN = THESIS_ROOT / "devign_full" / "after_ggnn"
OUT_DIR    = Path(__file__).parent / "preds"
OUT_DIR.mkdir(exist_ok=True)

LAMBDA1 = 0.5; LAMBDA2 = 0.001; ALPHA = 0.5
HIDDEN_DIM = 256; DROPOUT = 0.2; NUM_LAYERS = 1
BATCH_SIZE = 128; MAX_EPOCHS = 100; PATIENCE = 5
N_TRIALS   = 10
CUDA_DEVICE = -1  # force CPU — MetricLearningModel is lightweight and trainer.py calls .numpy() on tensors

CONDITIONS_FILES = {
    "clean":    "test_GGNNinput_graph.json",
    "ren":      "obf_identifier_test_GGNNinput_graph.json",
    "dead":     "obf_deadcode_test_GGNNinput_graph.json",
    "cf":       "obf_controlflow_test_GGNNinput_graph.json",
    "compound": "obf_compound_test_GGNNinput_graph.json",
}


def load_embeddings(path):
    data     = json.load(open(path))
    features = np.array([d["graph_feature"] for d in data], dtype=np.float32)
    targets  = np.array([d["target"] for d in data], dtype=np.int64)
    return features, targets


def main():
    print(f"Device: {'cuda:'+str(CUDA_DEVICE) if CUDA_DEVICE != -1 else 'cpu'}", flush=True)

    avail = {}
    for cond, fname in CONDITIONS_FILES.items():
        path = AFTER_GGNN / fname
        if path.exists():
            avail[cond] = load_embeddings(path)
            print(f"  {cond}: {len(avail[cond][0])}", flush=True)
        else:
            print(f"  SKIP {cond}: not found", flush=True)

    X_train, Y_train = load_embeddings(AFTER_GGNN / "train_GGNNinput_graph.json")
    X_valid, Y_valid = load_embeddings(AFTER_GGNN / "valid_GGNNinput_graph.json")
    X_pool = np.concatenate([X_train, X_valid])
    Y_pool = np.concatenate([Y_train, Y_valid])
    print(f"  pool: {len(X_pool)}", flush=True)

    true_labels = avail["clean"][1].tolist()
    result = {"model": "reveal", "conditions": list(avail.keys()),
              "true_labels": true_labels, "seeds": {}}

    for trial in range(N_TRIALS):
        seed = 1000 + trial * 7
        np.random.seed(seed); torch.manual_seed(seed)
        train_X, _, train_Y, _ = train_test_split(X_pool, Y_pool, test_size=0.2, random_state=seed)
        print(f"\nTrial {trial+1}/{N_TRIALS}  train={len(train_X)}", flush=True)

        input_dim = train_X.shape[1]
        model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=HIDDEN_DIM, aplha=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2, dropout_p=DROPOUT, num_layers=NUM_LAYERS)
        optimizer = Adam(model.parameters())
        if CUDA_DEVICE != -1:
            model.cuda(device=CUDA_DEVICE)

        # build dataset with clean test for training
        dataset = DataSet(BATCH_SIZE, input_dim)
        clean_X, clean_Y = avail["clean"]
        for x, y in zip(train_X, train_Y):
            split = 'valid' if np.random.uniform() <= 0.1 else 'train'
            dataset.add_data_entry(x.tolist(), int(y), split)
        for x, y in zip(clean_X, clean_Y):
            dataset.add_data_entry(x.tolist(), int(y), 'test')
        dataset.initialize_dataset(balance=True, output_buffer=None)

        reveal_train(model=model, dataset=dataset, optimizer=optimizer,
                     num_epochs=MAX_EPOCHS, max_patience=PATIENCE,
                     cuda_device=CUDA_DEVICE, output_buffer=None)

        trial_preds = {}
        for cond, (X_cond, Y_cond) in avail.items():
            dataset.clear_test_set()
            for x, y in zip(X_cond, Y_cond):
                dataset.add_data_entry(x.tolist(), int(y), 'test')
            n = dataset.initialize_test_batches()
            preds = reveal_predict(model=model,
                                   iterator_function=dataset.get_next_test_batch,
                                   _batch_count=n, cuda_device=CUDA_DEVICE)
            trial_preds[cond] = preds.tolist()
            n = min(len(true_labels), len(preds))
            print(f"  {cond}: F1={f1_score(true_labels[:n], preds[:n], zero_division=0)*100:.2f}% ({len(preds)} samples)",
                  flush=True)

        result["seeds"][str(seed)] = trial_preds
        del model

    out = OUT_DIR / "reveal_preds.json"
    with open(out, "w") as f: json.dump(result, f)
    print(f"Saved → {out}", flush=True)


if __name__ == "__main__":
    main()
