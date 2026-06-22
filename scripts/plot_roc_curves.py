#!/usr/bin/env python3
"""
plot_roc_curves.py — Generate publication-quality ROC curves for all baseline models.

Reads:  ~/thesis/devign_full/roc_data{_dataset}.json
Writes: ~/thesis/figures/roc_curves{_dataset}_all.pdf/png
        ~/thesis/figures/roc_curves{_dataset}_grouped.pdf/png

Usage:
    python ~/thesis/plot_roc_curves.py                     # Devign
    python ~/thesis/plot_roc_curves.py --dataset bigvul
    python ~/thesis/plot_roc_curves.py --dataset diversevul
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="devign", choices=["devign","bigvul","diversevul"])
args = parser.parse_args()

DATASET    = args.dataset
ROC_SUFFIX = "" if DATASET == "devign" else f"_{DATASET}"
ROC_FILE   = Path.home() / f"thesis/devign_full/roc_data{ROC_SUFFIX}.json"
FIG_DIR    = Path.home() / "thesis/figures"
FIG_DIR.mkdir(exist_ok=True)

DATASET_TITLES = {
    "devign":    "Devign Clean Test Set",
    "bigvul":    "Big-Vul Clean Test Set",
    "diversevul":"DiverseVul Balanced Test Set",
}

# ── Display names and grouping ────────────────────────────────────────────────
# Note: ANGLE, ReGVD, Vul-LMGGNN excluded — probability extraction unreliable
# (ANGLE: different tokenisation pipeline; ReGVD/LMGGNN: init incompatibilities)
# For BigVul/DiverseVul, CodeBERT-Aug was not evaluated; REVEAL is GNN-based (not GGNN LR)
MODEL_GROUPS = {
    "W-GNN": ["ECG RGCN", "VulGNN"],
    "Graph+LM": ["REVEAL", "Vul-LMGGNN"],
    "Transformer": ["CodeBERT", "CodeT5+", "CodeBERT-Aug"],
    "Non-neural": ["TF-IDF+LR", "CPG+LR"],
}

DEVIGN_GROUPS = {
    "W-GNN": ["ECG RGCN", "VulGNN"],
    "Graph+LM": ["REVEAL"],
    "Transformer": ["CodeBERT", "CodeT5+", "CodeBERT-Aug"],
    "Non-neural": ["TF-IDF+LR", "CPG+LR"],
}

CROSS_GROUPS = {
    "W-GNN": ["ECG RGCN", "VulGNN"],
    "Graph+LM": ["REVEAL"],
    "Transformer": ["CodeBERT", "CodeT5+"],
    "Non-neural": ["TF-IDF+LR"],
}

DISPLAY_NAMES = {
    "ECG RGCN":     "ECG RGCN",
    "ANGLE":        "ANGLE",
    "VulGNN":       "VulGNN",
    "REVEAL":       "REVEAL",
    "ReGVD":        "ReGVD",
    "Vul-LMGGNN":   "Vul-LMGGNN",
    "CodeBERT":     "CodeBERT",
    "CodeT5+":      "CodeT5+",
    "CodeBERT-Aug": "CodeBERT-Aug",
    "TF-IDF+LR":    "TF-IDF+LR",
    "CPG+LR":       "CPG+LR",
}

# ── Colors per group (colorblind-friendly) ────────────────────────────────────
GROUP_COLORS = {
    "W-GNN":       ["#1f77b4", "#17becf"],                  # blues
    "Graph+LM":    ["#2ca02c"],                             # green
    "Transformer": ["#d62728", "#ff9896", "#e377c2"],       # reds/pink
    "Non-neural":  ["#7f7f7f", "#c7c7c7"],                  # greys
}

LINESTYLES = ["-", "--", ":"]


def plot_all_roc(roc_data):
    """Single panel with all models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    active_groups = DEVIGN_GROUPS if DATASET == "devign" else CROSS_GROUPS
    for group, members in active_groups.items():
        colors = GROUP_COLORS[group]
        for i, m in enumerate(members):
            if m not in roc_data:
                continue
            d = roc_data[m]
            probs  = np.array(d["probs"])
            labels = np.array(d["labels"])
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            color = colors[i % len(colors)]
            ls    = LINESTYLES[i % len(LINESTYLES)]
            ax.plot(fpr, tpr, color=color, lw=1.6, linestyle=ls,
                    label=f'{DISPLAY_NAMES[m]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    dset_title = DATASET_TITLES.get(DATASET, DATASET)
    ax.set_title(f"ROC Curves — {dset_title}", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grouped_roc(roc_data):
    """2×2 grid: one panel per model family."""
    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    active_groups = DEVIGN_GROUPS if DATASET == "devign" else CROSS_GROUPS
    group_list = list(active_groups.items())
    positions  = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (group, members), (r, c) in zip(group_list, positions):
        ax = fig.add_subplot(gs[r, c])
        colors = GROUP_COLORS[group]
        for i, m in enumerate(members):
            if m not in roc_data:
                ax.plot([], [], label=f"{DISPLAY_NAMES[m]} (missing)")
                continue
            d = roc_data[m]
            fpr, tpr, _ = roc_curve(np.array(d["labels"]), np.array(d["probs"]))
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    lw=2.0, linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=f'{DISPLAY_NAMES[m]} ({roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_title(group, fontsize=12, fontweight='bold')
        ax.set_xlabel("FPR", fontsize=10)
        ax.set_ylabel("TPR", fontsize=10)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
        ax.grid(True, alpha=0.3)

    title = f"ROC Curves by Architectural Family — {DATASET_TITLES.get(DATASET, DATASET)}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def print_auc_table(roc_data, groups):
    print(f"\n{'Model':<16}  AUC-ROC")
    print("-" * 28)
    for group, members in groups.items():
        print(f"  [{group}]")
        for m in members:
            if m in roc_data:
                a = roc_data[m]["auc"]
                print(f"  {DISPLAY_NAMES[m]:<14}  {a:.4f}")
            else:
                print(f"  {DISPLAY_NAMES[m]:<14}  MISSING")


def main():
    if not ROC_FILE.exists():
        print(f"ERROR: {ROC_FILE} not found.")
        return

    roc_data = json.load(open(ROC_FILE))
    print(f"Loaded ROC data for {len(roc_data)} models: {list(roc_data.keys())}")

    active_groups = DEVIGN_GROUPS if DATASET == "devign" else CROSS_GROUPS
    print_auc_table(roc_data, active_groups)

    pfx = f"roc_curves{ROC_SUFFIX}"

    # Single-panel all-models figure
    fig1 = plot_all_roc(roc_data)
    out1_pdf = FIG_DIR / f"{pfx}_all.pdf"
    out1_png = FIG_DIR / f"{pfx}_all.png"
    fig1.savefig(out1_pdf, bbox_inches="tight")
    fig1.savefig(out1_png, bbox_inches="tight", dpi=200)
    print(f"\nSaved: {out1_pdf}")
    print(f"Saved: {out1_png}")

    # 2×2 grouped figure
    fig2 = plot_grouped_roc(roc_data)
    out2_pdf = FIG_DIR / f"{pfx}_grouped.pdf"
    out2_png = FIG_DIR / f"{pfx}_grouped.png"
    fig2.savefig(out2_pdf, bbox_inches="tight")
    fig2.savefig(out2_png, bbox_inches="tight", dpi=200)
    print(f"Saved: {out2_pdf}")
    print(f"Saved: {out2_png}")

    plt.close("all")


if __name__ == "__main__":
    main()
