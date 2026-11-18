"""Full-width (figure*) Delta-F1 heatmaps for Devign, DiverseVul, Big-Vul and PrimeVul, one panel each, from the same
stats files as pipeline/report/analyses.py:heatmaps(). Bold = significant (as in Tables III-V).
Each panel has its own symmetric colour scale (Big-Vul effects are ~4x Devign's); the DiverseVul scale is
capped at the Big-Vul range so that ANGLE's large drops do not wash out the other cells (marked on its colour bar).
Run with RERUN_ROOT pointing at a results root whose stats/ holds all four *_valid.json files, from the repository root."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from pipeline.models.common import CONDITIONS
from pipeline.report.analyses import MAIN, NAMES, stat_rows
from pipeline.report.tables import COND_SHORT

OUT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=1.0, help="figure size relative to full text width (e.g. 0.8)")
ap.add_argument("--out", default="fig_heatmaps_all4.pdf")
ap.add_argument("--font", help="font family for all text (default: matplotlib default)")
ap.add_argument("--sig-weight", default="bold", help="font weight for significant cells")
ap.add_argument("--black-text", action="store_true", help="all cell labels in black")
args = ap.parse_args()
S = args.scale
# Fonts are set for the printed size: shrinking the figure keeps the numbers readable by
# shrinking them less than the figure (cell labels never go below 5 pt in print).
F = ({"base": 6.5, "cell": 5.2, "title": 7.5, "cbar": 5.5} if S >= 1.0 else
     {"base": 6.0, "cell": 5.0, "title": 7.0, "cbar": 5.0} if S >= 0.75 else
     {"base": 5.0, "cell": 4.2, "title": 6.0, "cbar": 4.2})
plt.rcParams.update({"font.size": F["base"], "pdf.fonttype": 42, "ps.fonttype": 42})
if args.font:
    plt.rcParams["font.family"] = args.font

DS = [("devign", "Devign"), ("diversevul", "DiverseVul"), ("bigvul", "Big-Vul"), ("primevul", "PrimeVul")]
LIM = {}
_b = stat_rows("bigvul"); LIM["diversevul"] = np.ceil(max(abs(r["delta_f1"]) for (m, c), r in _b.items() if m in MAIN))
fig, axes = plt.subplots(2, 2, figsize=(7.16 * S, 4.6 * S))  # IEEE two-column text width, 2x2 so cell labels fit
for k, (ax, (ds, nice)) in enumerate(zip(axes.flat, DS)):
    st = stat_rows(ds)
    ms = [m for m in MAIN if (m, "identifier") in st]
    a = np.array([[st[(m, c)]["delta_f1"] if (m, c) in st else np.nan for c in CONDITIONS] for m in ms])
    lim = min(np.nanmax(np.abs(a)), LIM.get(ds, np.inf))
    clipped = np.nanmax(np.abs(a)) > lim
    im = ax.imshow(a, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(CONDITIONS)), COND_SHORT if k >= 2 else [], rotation=90)
    ax.set_yticks(range(len(ms)), [NAMES.get(m, m) for m in ms] if k % 2 == 0 else [])
    ax.tick_params(length=0, pad=1.5)
    for i, m in enumerate(ms):
        for j, c in enumerate(CONDITIONS):
            if not np.isnan(a[i, j]):
                ax.text(j, i, ("0.0" if round(a[i, j], 1) == 0 else f"{a[i, j]:+.1f}"), ha="center", va="center", fontsize=F["cell"],
                        fontweight=args.sig_weight if st[(m, c)].get("significant") else "normal",
                        color="black" if args.black_text or abs(a[i, j]) <= 0.6 * lim else "white")
    ax.set_title(nice, fontsize=F["title"], pad=3 * S)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="both" if clipped else "neither")
    cb.ax.tick_params(labelsize=F["cbar"], length=2 * S)
    if k % 2 == 1:
        cb.set_label("$\\Delta$F1 (pp)", fontsize=F["base"])
    assert ms == [m for m in MAIN if (m, "identifier") in stat_rows(DS[0][0])], "row order differs between panels"
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / args.out)
print("wrote", OUT / args.out)
