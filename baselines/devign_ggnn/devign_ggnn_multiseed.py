#!/usr/bin/env python3
"""Devign GGNN 5-seed multiseed evaluation on Devign dataset."""

import copy, json, random, sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR    = THESIS_ROOT / "devign_full" / "devign_input"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_ggnn_multiseed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from data_loader.dataset import DataSet, DataEntry
from modules.model import DevignModel
from trainer import train, evaluate_metrics

SEEDS        = [42, 1337, 7, 100, 999]
FEATURE_SIZE = 169
GRAPH_EMBED  = 200
NUM_STEPS    = 6
BATCH_SIZE   = 128
MAX_STEPS    = 1000
DEV_EVERY    = 10
MAX_PATIENCE = 20
LR           = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CONDITIONS = {
    "original":    "originals_train/test_GGNNinput.json",
    "identifier":  "obf_identifier_test/test_GGNNinput.json",
    "deadcode":    "obf_deadcode_test/test_GGNNinput.json",
    "controlflow": "obf_controlflow_test/test_GGNNinput.json",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_extra_test_examples(dataset, json_path):
    """Load a test JSON into DataEntry objects sharing dataset's edge_types."""
    examples = []
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        ex = DataEntry(
            datset=dataset,
            num_nodes=len(entry[dataset.n_ident]),
            features=entry[dataset.n_ident],
            edges=entry[dataset.g_ident],
            target=entry[dataset.l_ident][0][0],
        )
        examples.append(ex)
    return examples


def eval_on_examples(model, dataset, examples, loss_fn, threshold):
    dataset.test_examples = examples
    n_batches = dataset.initialize_test_batch()
    _, acc, pr, rc, f1 = evaluate_metrics(
        model=model,
        loss_function=loss_fn,
        num_batches=n_batches,
        data_iter=dataset.get_next_test_batch,
        device=DEVICE,
        threshold=threshold,
    )
    return {"f1": round(f1, 2), "pr": round(pr, 2),
            "rc": round(rc, 2), "acc": round(acc, 2)}


def train_one_seed(seed, dataset, test_sets, pos_weight):
    set_seed(seed)
    print(f"\n{'='*55}\n  Devign GGNN  Seed {seed}\n{'='*55}", flush=True)

    model = DevignModel(
        input_dim=FEATURE_SIZE,
        output_dim=GRAPH_EMBED,
        max_edge_types=dataset.max_edge_type,
        num_steps=NUM_STEPS,
    )
    loss_fn   = BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = Adam(model.parameters(), lr=LR)
    ckpt_path = str(CKPT_DIR / f"devign_ggnn_seed{seed}.pt")

    model, threshold = train(
        model=model,
        dataset=dataset,
        max_steps=MAX_STEPS,
        dev_every=DEV_EVERY,
        loss_function=loss_fn,
        optimizer=optimizer,
        save_path=ckpt_path,
        log_every=50,
        max_patience=MAX_PATIENCE,
        device=DEVICE,
    )

    result = {"seed": seed}
    for cond, examples in test_sets.items():
        metrics = eval_on_examples(model, dataset, examples, loss_fn, threshold)
        result[cond] = metrics
        print(f"  {cond:<20} F1={metrics['f1']:.2f}%  Pr={metrics['pr']:.2f}%  Rc={metrics['rc']:.2f}%", flush=True)
    return result


def main():
    print(f"Device: {DEVICE}  Seeds: {SEEDS}", flush=True)
    print("Loading dataset (train+valid+original test)...", flush=True)

    dataset = DataSet(
        train_src=str(DATA_DIR / "originals_train/train_GGNNinput.json"),
        valid_src=str(DATA_DIR / "originals_train/valid_GGNNinput.json"),
        test_src=str(DATA_DIR / "originals_train/test_GGNNinput.json"),
        batch_size=BATCH_SIZE,
        n_ident="node_features",
        g_ident="graph",
        l_ident="targets",
    )

    pos_count = sum(e.target for e in dataset.train_examples)
    neg_count = len(dataset.train_examples) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count])
    print(f"  train={len(dataset.train_examples)} (pos={pos_count})  valid={len(dataset.valid_examples)}", flush=True)

    print("Loading test conditions...", flush=True)
    original_test = list(dataset.test_examples)
    test_sets = {"original": original_test}
    for cond, relpath in TEST_CONDITIONS.items():
        if cond == "original":
            continue
        path = DATA_DIR / relpath
        if path.exists():
            test_sets[cond] = load_extra_test_examples(dataset, path)
            print(f"  {cond}: {len(test_sets[cond])} examples", flush=True)

    all_results = [train_one_seed(s, dataset, test_sets, pos_weight) for s in SEEDS]

    conds = list(test_sets.keys())
    agg = {"n_seeds": len(SEEDS), "seeds": SEEDS, "model": "Devign GGNN", "dataset": "Devign"}
    for cond in conds:
        f1s = [r[cond]["f1"] for r in all_results if cond in r]
        agg[cond] = {
            "f1_mean": round(float(np.mean(f1s)), 2),
            "f1_std":  round(float(np.std(f1s)),  2),
            "all_f1":  f1s,
        }
    base = agg["original"]["f1_mean"]
    for cond in conds[1:]:
        agg[cond]["delta_f1"] = round(agg[cond]["f1_mean"] - base, 2)

    out = RESULTS_DIR / "devign_ggnn_multiseed_results.json"
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved → {out}", flush=True)
    print(f"  original  F1={agg['original']['f1_mean']:.2f}±{agg['original']['f1_std']:.2f}")
    for cond in conds[1:]:
        d = agg[cond]
        print(f"  {cond:<20} F1={d['f1_mean']:.2f}±{d['f1_std']:.2f}  Δ={d['delta_f1']:+.2f}")


if __name__ == "__main__":
    main()
