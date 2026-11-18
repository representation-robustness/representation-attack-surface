"""DeepWukong figures for the paper (fig:deepwukong_combined), regenerated from the RERUN_ROOT results so they
match Table X. Same look as the other figures (colours, line styles, markers, labels), sized for the
subfigure slots in the paper (0.52 / 0.47 of a 7.16 in figure*, ~1.75 in tall) so fonts print at ~6.5 pt.

  roc_deepwukong_obfuscation.pdf  mean ROC over five seeds (vertical averaging on a common FPR grid);
                                  legend AUC = mean of the per-seed AUCs, as in results/stats/deepwukong.json.
                                  Clean: all clean test XFGs. Transformed: the paired XFGs of Table X
                                  (found in both forms, key line in a validly transformed function).
  fig_dwk_breakdown.pdf           bars = XFGs per condition (Table X column), line = F1 (Table X column).

Run with RERUN_ROOT=<rerun root> from the repository root."""
import gzip
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path.cwd()))
from pipeline.deepwukong.common import ROOT, work
from pipeline.deepwukong.report import load_index

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

runs = [json.load(gzip.open(f)) for f in sorted(glob.glob(str(ROOT / "results/preds/deepwukong/deepwukong/run_*.json.gz")))]
assert len(runs) == 5, len(runs)
clean_idx = load_index("clean")
GRID = np.linspace(0, 1, 2001)


def scope(c):
    idx = load_index(c)
    return {i for i, x in idx.items() if x["found"] and x["valid_scope"] and clean_idx.get(i, {}).get("found")}


def mean_roc(c):
    sc = None if c == "clean" else scope(c)
    tprs, aucs, ns = [], [], set()
    for r in runs:
        cl, tr = r["test"]["clean"], r["test"][c]
        yc = dict(zip(cl["ids"], cl["y"]))
        pt = dict(zip(tr["ids"], tr["p"]))
        ids = sorted(pt if sc is None else sc & yc.keys() & pt.keys())
        y, p = np.array([yc[i] for i in ids]), np.array([pt[i] for i in ids])
        fpr, tpr, _ = roc_curve(y, p)
        t = np.interp(GRID, fpr, tpr); t[0] = 0.0
        tprs.append(t); aucs.append(roc_auc_score(y, p)); ns.add(len(ids))
    m = np.mean(tprs, axis=0); m[-1] = 1.0
    return m, float(np.mean(aucs)), ns


# ---------------- (a) ROC ----------------
CURVES = [("clean", "Clean", "tab:blue", "-", 1.2),
          ("identifier", "Identifier rename", "tab:red", "--", 1.0),
          ("deadcode", "Dead-code", "tab:orange", "--", 1.0),
          ("controlflow", "Control-flow", "tab:green", "--", 1.0),
          ("compound", "Compound", "tab:purple", "--", 1.0)]
stats = json.load(open(ROOT / "results/stats/deepwukong.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})
fig, ax = plt.subplots(figsize=(3.6, 1.75))
ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Random (AUC=0.50)")
curves = {}
for c, name, col, ls, lw in CURVES:
    m, auc, ns = mean_roc(c)
    if c != "clean":  # same numbers as the stats file / Table X computation
        assert abs(100 * auc - stats["conditions"][c]["paired"]["roc_auc"]["trans_mean"]) < 1e-6, c
    curves[c] = (m, col, ls, lw)
    ax.plot(GRID, m, ls, color=col, lw=lw, label=f"{name} (AUC={auc:.3f})")
    print(f"{c:12s} AUC={auc:.4f} n={sorted(ns)}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate", labelpad=1); ax.set_ylabel("True Positive Rate", labelpad=1)
ax.set_title("DeepWukong ROC Under Obfuscation\n(SARD CWE-119)", fontweight="bold", fontsize=7, pad=3)
ax.grid(alpha=0.3, lw=0.5)
ax.tick_params(length=2, pad=1.5)
ax.set_aspect("equal")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.8, frameon=True, handlelength=2.2)
# the curves sit in the top-left corner (AUC > 0.95): zoomed inset so they can be told apart
ins = ax.inset_axes([0.42, 0.10, 0.52, 0.52])
for c, (m, col, ls, lw) in curves.items():
    ins.plot(GRID, m, ls, color=col, lw=lw * 0.9)
ins.set_xlim(0, 0.2); ins.set_ylim(0.6, 1.0)
ins.set_xticks([0, 0.1, 0.2]); ins.set_yticks([0.6, 0.8, 1.0])
ins.tick_params(labelsize=4.8, length=1.5, pad=1)
ins.grid(alpha=0.3, lw=0.4)
ind = ax.indicate_inset_zoom(ins, edgecolor="0.4", lw=0.5)
for cn in ind.connectors:  # box only; connector lines cut across the curves
    cn.set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "roc_deepwukong_obfuscation.pdf")
plt.close(fig)

# ---------------- (b) XFGs and F1 ----------------
# Table X: XFG counts (paired scope) and F1 per condition, mean over seeds
LAB = ["Original", "Rename", "Dead", "CF", "R+D", "R+CF", "D+CF", "Cmp"]
CONDS = ["identifier", "deadcode", "controlflow", "ren_dead", "ren_cf", "dead_cf", "compound"]
n = [stats["clean_shipped"][0]["n"]] + [stats["conditions"][c]["paired"]["n"] for c in CONDS]
f1 = [np.mean([r["f1"] for r in stats["clean_shipped"]])] + [stats["conditions"][c]["paired"]["f1"]["trans_mean"] for c in CONDS]
print("XFGs", n); print("F1", [round(x, 2) for x in f1])
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 6.5})
RED = "#c44e52"
fig, ax = plt.subplots(figsize=(3.37, 1.8))
x = np.arange(len(LAB))
ax.bar(x, n, color="#a6c8e6", width=0.8, label="XFG count", zorder=2)
ax.set_ylabel("XFGs evaluated", labelpad=2)
ax.set_ylim(0, 12000)
ax.set_xticks(x, LAB, rotation=20, ha="right")
ax.tick_params(length=2, pad=1.5)
ax2 = ax.twinx()
ax2.plot(x, f1, "-o", color=RED, lw=1.1, ms=2.6, label="F1 (%)", zorder=3)
for xi, v in zip(x, f1):
    ax2.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 3.5), ha="center", fontsize=5.2, color=RED)
ax2.set_ylim(0, 120)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_ylabel("F1 score (%)", color=RED, labelpad=2)
ax2.tick_params(axis="y", colors=RED, length=2, pad=1.5)
ax.set_title("DeepWukong on SARD CWE-119: XFG count and F1 per condition", fontsize=6.8, pad=3)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=5.5, frameon=True)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_dwk_breakdown.pdf")
print("wrote", OUT / "roc_deepwukong_obfuscation.pdf", OUT / "fig_dwk_breakdown.pdf")
