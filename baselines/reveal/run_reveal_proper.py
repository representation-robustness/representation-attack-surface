#!/usr/bin/env python3
"""
REVEAL full pipeline: representation learning (triplet loss + SMOTE) on GGNN embeddings.

This replicates REVEAL exactly as described in Chakraborty et al. 2022:
  - Input: 200-dim GGNN graph embeddings (precomputed by Devign GGNN)
  - SMOTE oversampling of the minority class (balance=True)
  - MetricLearningModel: triplet loss + cross-entropy + L2 regularization
    - lambda1=0.5 (triplet weight), lambda2=0.001 (L2), alpha=0.5 (margin)
    - hidden_dim=256, dropout=0.2, num_layers=1
  - Adam optimizer, max_patience=5 (as in api_test.py), batch_size=128
  - Best checkpoint by val F1

Evaluation: our fixed train/val/test splits for all 4 conditions.
Reports mean ± std over N_TRIALS (matching paper's 30-trial protocol).

Usage (from thesis root):
  source /home/jesse/venvs/reveal310/bin/activate
  CUDA_VISIBLE_DEVICES=1 python baselines/reveal_ggnn/run_reveal_proper.py
"""

import copy
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- path setup so we can import REVEAL's own modules ---
SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
REVEAL_RL   = THESIS_ROOT / "data/raw/ReVeal/Vuld_SySe/representation_learning"
sys.path.insert(0, str(REVEAL_RL))

from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train as reveal_train, evaluate as reveal_evaluate
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Paths to GGNN embeddings
# ---------------------------------------------------------------------------
AFTER_GGNN = THESIS_ROOT / "devign_full" / "after_ggnn"

CONDITIONS = {
    "originals":   AFTER_GGNN / "test_GGNNinput_graph.json",
    "identifier":  AFTER_GGNN / "obf_identifier_test_GGNNinput_graph.json",
    "deadcode":    AFTER_GGNN / "obf_deadcode_test_GGNNinput_graph.json",
    "controlflow": AFTER_GGNN / "obf_controlflow_test_GGNNinput_graph.json",
}

# ---------------------------------------------------------------------------
# REVEAL hyperparameters (matching api_test.py)
# ---------------------------------------------------------------------------
LAMBDA1    = 0.5
LAMBDA2    = 0.001
ALPHA      = 0.5
HIDDEN_DIM = 256
DROPOUT    = 0.2
NUM_LAYERS = 1
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE   = 5     # api_test.py uses max_patience=5
N_TRIALS   = 10    # paper does 30; we do 10 for time

CUDA_DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    features = np.array([d["graph_feature"] for d in data], dtype=np.float32)
    targets  = np.array([d["target"] for d in data], dtype=np.int64)
    return features, targets


def load_all_splits():
    """Load train+val combined (REVEAL uses random re-splits each trial)."""
    train_f, train_t = load_embeddings(AFTER_GGNN / "train_GGNNinput_graph.json")
    valid_f, valid_t = load_embeddings(AFTER_GGNN / "valid_GGNNinput_graph.json")
    # Combine train+val — api_test.py combines all splits then re-splits randomly
    X = np.concatenate([train_f, valid_f], axis=0)
    Y = np.concatenate([train_t, valid_t], axis=0)
    return X, Y


def load_condition(json_path: Path):
    return load_embeddings(json_path)


# ---------------------------------------------------------------------------
# Single trial: train on train_X/train_Y, evaluate on fixed test sets
# ---------------------------------------------------------------------------

def run_trial(train_X, train_Y, test_sets: dict, seed: int):
    """One trial matching REVEAL's RepresentationLearningModel.train() call."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = train_X.shape[1]

    model = MetricLearningModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        aplha=ALPHA,
        lambda1=LAMBDA1,
        lambda2=LAMBDA2,
        dropout_p=DROPOUT,
        num_layers=NUM_LAYERS,
    )
    optimizer = Adam(model.parameters())
    if CUDA_DEVICE != -1:
        model.cuda(device=CUDA_DEVICE)

    # Build DataSet — exactly as RepresentationLearningModel.train() does it
    dataset = DataSet(BATCH_SIZE, input_dim)
    for x, y in zip(train_X, train_Y):
        # 10% of train goes to internal validation (api's approach)
        if np.random.uniform() <= 0.1:
            dataset.add_data_entry(x.tolist(), int(y), 'valid')
        else:
            dataset.add_data_entry(x.tolist(), int(y), 'train')

    # Add originals test set for monitoring during training (not used for selection)
    orig_X, orig_Y = test_sets["originals"]
    for x, y in zip(orig_X, orig_Y):
        dataset.add_data_entry(x.tolist(), int(y), 'test')

    # SMOTE balance + initialize batches
    dataset.initialize_dataset(balance=True, output_buffer=None)

    reveal_train(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        num_epochs=MAX_EPOCHS,
        max_patience=PATIENCE,
        cuda_device=CUDA_DEVICE,
        output_buffer=None,
    )

    # Evaluate on all conditions
    results = {}
    for name, (X_cond, Y_cond) in test_sets.items():
        dataset.clear_test_set()
        for x, y in zip(X_cond, Y_cond):
            dataset.add_data_entry(x.tolist(), int(y), 'test')
        n_batches = dataset.initialize_test_batches()
        acc, pr, rc, f1 = reveal_evaluate(
            model=model,
            iterator_function=dataset.get_next_test_batch,
            _batch_count=n_batches,
            cuda_device=CUDA_DEVICE,
            output_buffer=None,
        )
        results[name] = {"acc": acc, "pr": pr, "rc": rc, "f1": f1}

    return results


# ---------------------------------------------------------------------------
# Main: N_TRIALS trials, each with a fresh 80/20 train/test split
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {'cuda:' + str(CUDA_DEVICE) if CUDA_DEVICE != -1 else 'cpu'}", flush=True)
    print("Loading GGNN embeddings...", flush=True)

    # Load combined train+val (matching api_test.py which uses all 3 parts)
    X_pool, Y_pool = load_all_splits()
    print(f"Train pool: {len(X_pool)} samples "
          f"(pos={Y_pool.sum()}, neg={len(Y_pool)-Y_pool.sum()})", flush=True)

    # Load fixed condition test sets
    test_sets = {}
    for name, path in CONDITIONS.items():
        X_cond, Y_cond = load_condition(path)
        test_sets[name] = (X_cond, Y_cond)
        print(f"  {name}: {len(X_cond)} samples "
              f"(pos={Y_cond.sum()}, neg={len(Y_cond)-Y_cond.sum()})")

    print(f"\nRunning {N_TRIALS} trials (lambda1={LAMBDA1}, patience={PATIENCE})...\n",
          flush=True)

    all_results = {name: [] for name in CONDITIONS}

    for trial in range(N_TRIALS):
        seed = 1000 + trial * 7
        # Follow api_test.py: random 80/20 split of the pool each trial
        train_X, _, train_Y, _ = train_test_split(
            X_pool, Y_pool, test_size=0.2, random_state=seed
        )
        print(f"Trial {trial+1}/{N_TRIALS} | train={len(train_X)} "
              f"(pos={train_Y.sum()}, neg={len(train_Y)-train_Y.sum()})",
              flush=True)

        trial_results = run_trial(train_X, train_Y, test_sets, seed)

        for name, r in trial_results.items():
            all_results[name].append(r)
            if name == "originals":
                print(f"  originals -> F1={r['f1']:.2f}%  Rc={r['rc']:.2f}%", flush=True)

    # Summarise
    print("\n" + "="*70)
    print("=== REVEAL Results (mean ± std over %d trials) ===" % N_TRIALS)
    print("="*70)

    summary = {}
    base_f1 = None
    for name in CONDITIONS:
        f1s = [r["f1"] for r in all_results[name]]
        rcs = [r["rc"] for r in all_results[name]]
        prs = [r["pr"] for r in all_results[name]]
        accs = [r["acc"] for r in all_results[name]]

        mean_f1 = np.mean(f1s)
        std_f1  = np.std(f1s)
        mean_rc = np.mean(rcs)
        mean_pr = np.mean(prs)
        mean_acc = np.mean(accs)

        delta = (mean_f1 - base_f1) if base_f1 is not None else 0.0
        if base_f1 is None:
            base_f1 = mean_f1

        delta_str = f"  ΔF1={delta:+.2f}%" if name != "originals" else ""
        print(f"{name:15s}  F1={mean_f1:.2f}±{std_f1:.2f}%  "
              f"Acc={mean_acc:.2f}%  Pr={mean_pr:.2f}%  Rc={mean_rc:.2f}%{delta_str}")

        summary[name] = {
            "f1_mean": mean_f1, "f1_std": std_f1,
            "acc_mean": mean_acc, "pr_mean": mean_pr, "rc_mean": mean_rc,
            "delta_f1": delta,
            "all_f1": f1s,
        }

    # Save
    out = THESIS_ROOT / "devign_full" / "reveal_proper_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
