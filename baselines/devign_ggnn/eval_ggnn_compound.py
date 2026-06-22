#!/usr/bin/env python3
"""Eval Dev.GGNN on compound test using saved checkpoints.

By renaming invariance, ΔD+CF = ΔCmp (compound = ren+dead+cf, ren has zero effect).
Fills the pairwise columns for Dev.GGNN Devign table row.
"""
import copy, json, sys
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
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_extra_examples(dataset, json_path):
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
    return {"f1": round(f1, 2), "pr": round(pr, 2), "rc": round(rc, 2), "acc": round(acc, 2)}


def main():
    print(f"Device: {DEVICE}", flush=True)

    print("Loading dataset...", flush=True)
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
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    print(f"  max_edge_type={dataset.max_edge_type}", flush=True)

    print("Loading compound test...", flush=True)
    compound_path = DATA_DIR / "obf_compound_test" / "test_GGNNinput.json"
    compound_examples = load_extra_examples(dataset, compound_path)
    original_examples = list(dataset.test_examples)
    print(f"  compound: {len(compound_examples)} examples", flush=True)

    compound_f1s = []
    original_f1s = []

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"devign_ggnn_seed{seed}.pt"
        print(f"\nSeed {seed}...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        threshold = ckpt["best_threshold"]
        print(f"  threshold={threshold:.2f}", flush=True)

        model = DevignModel(
            input_dim=FEATURE_SIZE,
            output_dim=GRAPH_EMBED,
            max_edge_types=dataset.max_edge_type,
            num_steps=NUM_STEPS,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()

        orig_m = eval_on_examples(model, dataset, original_examples, loss_fn, threshold)
        cmp_m  = eval_on_examples(model, dataset, compound_examples, loss_fn, threshold)
        print(f"  original  F1={orig_m['f1']:.2f}%", flush=True)
        print(f"  compound  F1={cmp_m['f1']:.2f}%  Δ={cmp_m['f1']-orig_m['f1']:+.2f}pp", flush=True)
        original_f1s.append(orig_m["f1"])
        compound_f1s.append(cmp_m["f1"])

    base = float(np.mean(original_f1s))
    cmp  = float(np.mean(compound_f1s))
    delta_cmp = round(cmp - base, 2)

    print("\n=== RESULTS ===")
    print(f"  original:    {base:.2f}±{float(np.std(original_f1s)):.2f}%")
    print(f"  compound:    {cmp:.2f}±{float(np.std(compound_f1s)):.2f}%  Δ={delta_cmp:+.2f}pp")
    print(f"\nBy renaming invariance:")
    print(f"  ΔD+CF = ΔCmp = {delta_cmp:+.2f}pp")
    print(f"  ΔR+D  = ΔDead = -0.03pp  (from existing results)")
    print(f"  ΔR+CF = ΔCF  = -0.17pp  (from existing results)")

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
        "note":     "identical to deadcode (renaming-invariant node features)",
    }
    existing["ren_cf"] = {
        "f1_mean":  existing["controlflow"]["f1_mean"],
        "f1_std":   existing["controlflow"]["f1_std"],
        "all_f1":   existing["controlflow"]["all_f1"],
        "delta_f1": cf_delta,
        "note":     "identical to controlflow (renaming-invariant node features)",
    }
    existing["dead_cf"] = {
        "f1_mean":  round(cmp, 2),
        "f1_std":   round(float(np.std(compound_f1s)), 2),
        "all_f1":   [round(v, 2) for v in compound_f1s],
        "delta_f1": delta_cmp,
        "note":     "identical to compound (renaming-invariant node features)",
    }

    out_path = RESULTS_DIR / "devign_ggnn_7cond_results.json"
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
