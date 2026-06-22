"""
Generate ΔF1 heatmaps for Devign, Big-Vul, and DiverseVul.
Aesthetic: clean sans-serif, seaborn heatmap, colorbar on right.
Output: ~/thesis/figures/ as PDF + PNG at 300 DPI.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

OUT = '/home/jesse/thesis/figures/'

plt.rcParams.update({
    'font.family': 'Ubuntu',
    'font.size':    9,
    'figure.dpi':  300,
    'savefig.dpi': 300,
})

COLS = ['Ren', 'Dead', 'CF', 'R+D', 'R+CF', 'D+CF', 'Cmp']


def fmt_val(v):
    if np.isnan(v):
        return '—'
    if v == 0.0:
        return '0.00'
    return f'{v:+.2f}'


def make_heatmap(data, row_labels, title, fname, vmax,
                 annot_fs=8.0, tick_fs=8.5, title_fs=10, cbar_fs=8.5, cbar_tick_fs=8):
    n_rows = len(row_labels)
    annot = np.vectorize(fmt_val)(data)

    df = pd.DataFrame(data, index=row_labels, columns=COLS)

    fig_h = max(3.5, 0.46 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    sns.heatmap(
        df,
        ax=ax,
        cmap='RdBu',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt='',
        linewidths=0.4,
        linecolor='#dddddd',
        cbar_kws={'label': 'ΔF1 (pp)', 'shrink': 0.75, 'aspect': 20},
        annot_kws={'fontsize': annot_fs, 'fontweight': 'medium'},
    )

    ax.set_title(title, fontsize=title_fs, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', length=0, labelsize=tick_fs)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Rotate x-labels to horizontal
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=tick_fs, fontweight='medium')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs, fontweight='medium')

    # Fix colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.set_label('ΔF1 (pp)', fontsize=cbar_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    plt.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        plt.savefig(f'{OUT}{fname}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓  {fname} saved')


# ════════════════════════════════════════════════════════════════════════════
# Devign  (12 systems)
# ════════════════════════════════════════════════════════════════════════════
devign_labels = [
    'ECG RGCN',
    'ANGLE',
    'VulGNN',
    'REVEAL',
    'Dev. GGNN†',
    'Vul-LMGGNN',
    'ReGVD',
    'CodeBERT',
    'CodeT5+',
    'CodeBERT-Aug',
    'TF-IDF+LR',
    'CPG+LR',
]
devign_data = np.array([
    #  Ren    Dead    CF     R+D    R+CF   D+CF   Cmp
    [  0.00, +0.61, -1.20, +0.61, -1.20, +0.56, +0.55],  # ECG RGCN
    [  0.00, +1.38, -1.19, +1.39, -1.19, +1.41, +1.41],  # ANGLE
    [  0.00, +1.24, +0.61, +1.24, +0.61, +1.49, +1.49],  # VulGNN
    [ -6.08, +1.59, -3.63, +1.33, +0.23, +1.19, +1.58],  # REVEAL
    [ -0.15, -0.03, -0.17, -0.03, -0.17, -0.48, -0.48],  # Dev. GGNN†
    [ -2.19, +0.61, +0.01, +0.49, -0.07, +0.27, +0.24],  # Vul-LMGGNN
    [ -1.85, +0.93, +0.93, +0.79, +1.67, +1.55, +1.57],  # ReGVD
    [ -1.90, -1.71, -1.93, -1.94, -2.19, -1.85, -1.96],  # CodeBERT
    [ -0.68, +0.64, +0.16, +0.67, +0.51, +0.79, +0.72],  # CodeT5+
    [ +0.08, -0.27, -0.57, -0.68, -0.52, -0.52, -0.85],  # CodeBERT-Aug
    [ -4.02, +0.28, +0.25, -3.22, -3.01, +0.43, -2.48],  # TF-IDF+LR
    [ -6.30, +6.11, -1.11, +6.11, -1.11, +4.36, +4.36],  # CPG+LR
])

make_heatmap(
    devign_data, devign_labels,
    title='Devign: ΔF1 under each transformation (pp)',
    fname='fig_devign_heatmap',
    vmax=7.0,
    annot_fs=13, tick_fs=13, title_fs=15, cbar_fs=13, cbar_tick_fs=12,
)

# ════════════════════════════════════════════════════════════════════════════
# Big-Vul  (10 systems)
# ════════════════════════════════════════════════════════════════════════════
bigvul_labels = [
    'ECG RGCN',
    'ANGLE',
    'VulGNN',
    'REVEAL',
    'Vul-LMGGNN',
    'ReGVD',
    'CodeBERT',
    'CodeT5+',
    'TF-IDF+LR',
    'CPG+LR',
]
bigvul_data = np.array([
    #  Ren    Dead     CF     R+D    R+CF    D+CF    Cmp
    [  0.00, -4.00,  -1.31, -3.99,  -1.32,  -4.70,  -4.71],
    [ +0.03, -5.66,  -0.68, -5.64,  -0.64,  -6.02,  -6.01],
    [ +0.01, -4.37,  -0.64, -4.38,  -0.64,  -4.96,  -4.97],
    [ -0.02, -1.26,  -0.52, -1.27,  -0.53,  -1.40,  -1.40],
    [ -1.05, -6.49,  -2.21, -8.12,  -3.78,  -8.28, -10.10],
    [ -5.57, -0.76,  -0.05, -7.56,  -5.67,  -2.21,  -7.54],
    [ -0.41,-18.81,  -2.58,-19.14,  -2.88, -20.94, -20.76],
    [ +0.72,-18.29,  -2.94,-17.56,  -2.34, -19.12, -18.83],
    [ -1.79, -3.72,  -1.42, -5.56,  -2.78,  -4.43,  -6.04],
    [  0.00, -3.82,  -1.20, -3.80,  -1.17,  -4.11,  -4.09],
])

make_heatmap(
    bigvul_data, bigvul_labels,
    title='Big-Vul: ΔF1 under each transformation (pp)',
    fname='fig_bigvul_heatmap',
    vmax=21.0,
    annot_fs=13, tick_fs=13, title_fs=15, cbar_fs=13, cbar_tick_fs=12,
)

# ════════════════════════════════════════════════════════════════════════════
# DiverseVul  (10 systems)
# ════════════════════════════════════════════════════════════════════════════
divvul_labels = [
    'ECG RGCN',
    'ANGLE',
    'VulGNN',
    'REVEAL',
    'Vul-LMGGNN',
    'ReGVD',
    'CodeBERT',
    'CodeT5+',
    'TF-IDF+LR',
    'CPG+LR',
]
divvul_data = np.array([
    #  Ren    Dead    CF     R+D    R+CF   D+CF   Cmp
    [  0.00, +0.84, -0.30, +0.84, -0.31, +0.47, +0.47],
    [  0.00, +3.28, +0.22, +3.28, +0.22, +2.99, +3.00],
    [  0.00, +2.56, -0.27, +2.56, -0.27, +2.42, +2.41],
    [  0.00, +0.54, -1.26, +0.54, -1.26, -0.06, -0.08],
    [ -1.14, -0.33, +0.64, -2.02, -0.40, -0.42, -2.30],
    [ -2.19, -0.97, -1.19, -3.26, -3.37, -1.14, -3.25],
    [ -0.16, -0.12, -0.44, -0.49, -0.52, -0.46, -0.67],
    [ -0.76, +0.47, -0.31, -0.39, -0.75, +0.46, -0.45],
    [ -2.67, +1.01, +0.37, -1.42, -1.91, +1.10, -1.28],
    [  0.00, +5.90, +0.91, +5.90, +0.91, +5.65, +5.65],
])

make_heatmap(
    divvul_data, divvul_labels,
    title='DiverseVul: ΔF1 under each transformation (pp)',
    fname='fig_diversevul_heatmap',
    vmax=6.0,
    annot_fs=13, tick_fs=13, title_fs=15, cbar_fs=13, cbar_tick_fs=12,
)

print(f'\nAll heatmaps saved to {OUT}')
