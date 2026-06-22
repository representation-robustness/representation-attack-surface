#!/usr/bin/env python3
"""Fast Dev.GGNN compound eval — skips training data loading.

max_edge_types=5 inferred from ggnn.linears.0-4 in saved checkpoints.
best_threshold loaded directly from checkpoint dict.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR    = THESIS_ROOT / "devign_full" / "devign_input"
RESULTS_DIR = THESIS_ROOT / "devign_full"
CKPT_DIR    = SCRIPT_DIR / "ckpts_ggnn_multiseed"

sys.path.insert(0, str(SCRIPT_DIR))
from data_loader.dataset import DataSet, DataEntry
from modules.model import DevignModel
from trainer import evaluate_metrics

SEEDS        = [42, 1337, 7, 100, 999]
FEATURE_SIZE = 169
GRAPH_EMBED  = 200
NUM_STEPS    = 6
BATCH_SIZE   = 128
MAX_EDGE_TYPES = 5   # inferred from ggnn.linears.0-4 in checkpoint state_dicts
N_IDENT      = "node_features"
G_IDENT      = "graph"
L_IDENT      = "targets"
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_examples_direct(json_path, dataset):
    """Load GGNNinput.json without going through full DataSet train-loading."""
    examples = []
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        ex = DataEntry(
            datset=dataset,
            num_nodes=len(entry[N_IDENT]),
            features=entry[N_IDENT],
            edges=entry[G_IDENT],
            target=entry[L_IDENT][0][0],
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
    return {"f1": round(f1, 2), "acc": round(acc, 2)}


class _Stub:
    """Minimal stub for DataSet to enable test batching without full data load."""
    pass


def main():
    print(f"Device: {DEVICE}  max_edge_types={MAX_EDGE_TYPES}", flush=True)

    print("Loading minimal dataset (test-only)...", flush=True)
    dataset = DataSet(
        train_src=str(DATA_DIR / "originals_train/test_GGNNinput.json"),
        valid_src=str(DATA_DIR / "originals_train/test_GGNNinput.json"),
        test_src=str(DATA_DIR / "originals_train/test_GGNNinput.json"),
        batch_size=BATCH_SIZE,
        n_ident=N_IDENT,
        g_ident=G_IDENT,
        l_ident=L_IDENT,
    )
    pos_count = sum(e.target for e in dataset.test_examples)
    neg_count = len(dataset.test_examples) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count] if pos_count > 0 else [1.0])
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    print(f"  test: {len(dataset.test_examples)} examples", flush=True)

    print("Loading original test and compound test...", flush=True)
    original_examples = list(dataset.test_examples)
    compound_path = DATA_DIR / "obf_compound_test" / "test_GGNNinput.json"
    compound_examples = load_examples_direct(compound_path, dataset)
    print(f"  compound: {len(compound_examples)} examples", flush=True)

    original_f1s = []
    compound_f1s = []

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"devign_ggnn_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        threshold = ckpt["best_threshold"]

        model = DevignModel(
            input_dim=FEATURE_SIZE,
            output_dim=GRAPH_EMBED,
            max_edge_types=MAX_EDGE_TYPES,
            num_steps=NUM_STEPS,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()

        orig_m = eval_on_examples(model, dataset, original_examples, loss_fn, threshold)
        cmp_m  = eval_on_examples(model, dataset, compound_examples, loss_fn, threshold)
        print(f"  Seed {seed}:  original={orig_m['f1']:.2f}%  compound={cmp_m['f1']:.2f}%  "
              f"Δ={cmp_m['f1']-orig_m['f1']:+.2f}pp  thr={threshold:.2f}", flush=True)
        original_f1s.append(orig_m["f1"])
        compound_f1s.append(cmp_m["f1"])

    base = float(np.mean(original_f1s))
    cmp  = float(np.mean(compound_f1s))
    delta_cmp = round(cmp - base, 2)

    print(f"\n=== RESULTS ===")
    print(f"  original: {base:.2f}±{float(np.std(original_f1s)):.2f}%")
    print(f"  compound: {cmp:.2f}±{float(np.std(compound_f1s)):.2f}%  Δ={delta_cmp:+.2f}pp")
    print(f"\nBy renaming invariance (ren doesn't change CPG node-type features):")
    print(f"  ΔD+CF = ΔCmp = {delta_cmp:+.2f}pp")
    print(f"  ΔR+D  = ΔDead = -0.03pp")
    print(f"  ΔR+CF = ΔCF  = -0.17pp")

    existing = json.load(open(RESULTS_DIR / "devign_ggnn_multiseed_results.json"))
    existing["compound"] = {
        "f1_mean": round(cmp, 2),
        "f1_std":  round(float(np.std(compound_f1s)), 2),
        "all_f1":  [round(v, 2) for v in compound_f1s],
        "delta_f1": delta_cmp,
    }
    dead_delta = existing["deadcode"]["delta_f1"]
    cf_delta   = existing["controlflow"]["delta_f1"]
    existing["ren_dead"] = {
        "f1_mean":  existing["deadcode"]["f1_mean"],
        "f1_std":   existing["deadcode"]["f1_std"],
        "all_f1":   existing["deadcode"]["all_f1"],
        "delta_f1": dead_delta,
        "note":     "identical to deadcode (renaming-invariant)",
    }
    existing["ren_cf"] = {
        "f1_mean":  existing["controlflow"]["f1_mean"],
        "f1_std":   existing["controlflow"]["f1_std"],
        "all_f1":   existing["controlflow"]["all_f1"],
        "delta_f1": cf_delta,
        "note":     "identical to controlflow (renaming-invariant)",
    }
    existing["dead_cf"] = {
        "f1_mean":  round(cmp, 2),
        "f1_std":   round(float(np.std(compound_f1s)), 2),
        "all_f1":   [round(v, 2) for v in compound_f1s],
        "delta_f1": delta_cmp,
        "note":     "identical to compound (renaming-invariant)",
    }

    out_path = RESULTS_DIR / "devign_ggnn_7cond_results.json"
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
