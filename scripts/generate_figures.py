"""
Generate all publication figures for the conference paper.
All data sourced from actual result files in ~/thesis/devign_full/.
Output: ~/thesis/figures/ at 300 DPI.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

OUT = '/home/jesse/thesis/figures/'

# ── Colour palette ───────────────────────────────────────────────────────────
FAM_COLORS = {
    'W-GNN':  '#1565C0',   # dark blue
    'G+LM':   '#BF360C',   # dark orange-red
    'Trans':  '#2E7D32',   # dark green
    'N-N':    '#6A1B9A',   # dark purple
}

REN_COLOR  = '#C62828'
DEAD_COLOR = '#1565C0'
CF_COLOR   = '#2E7D32'

DEV_COLOR  = '#1976D2'
BIG_COLOR  = '#E64A19'
DIV_COLOR  = '#388E3C'


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ΔF1 Grouped Bar Chart (Devign, all systems, 3 transforms)
# ════════════════════════════════════════════════════════════════════════════
def fig4_delta_bar_chart():
    # Data from paper Table 3 (multi-seed means)
    systems = [
        'ECG RGCN', 'ANGLE', 'VulGNN', 'REVEAL',
        'Vul-LMGGNN', 'ReGVD',
        'CodeBERT', 'CodeT5+', 'CB-Aug',
        'TF-IDF+LR', 'CPG+LR',
    ]
    families = ['W-GNN','W-GNN','W-GNN','W-GNN',
                'G+LM','G+LM',
                'Trans','Trans','Trans',
                'N-N','N-N']
    d_ren  = [-0.43,  0.00,  0.00, -6.08, -4.58, -1.95, -1.45, -0.68, +0.08, -4.02, -6.30]
    d_dead = [-0.14, +1.38, +1.24, +1.59, +2.38, +1.12, -0.56, +0.64, -0.27, +0.28, +6.11]
    d_cf   = [-0.33, -1.19, +0.61, -3.63, +1.86, +0.99, -1.02, +0.16, -0.57, +0.25, -1.11]

    n = len(systems)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars_ren  = ax.bar(x - w,   d_ren,  w, label='Identifier Renaming', color=REN_COLOR,  alpha=0.88, zorder=3)
    bars_dead = ax.bar(x,       d_dead, w, label='Dead-Code Insertion',  color=DEAD_COLOR, alpha=0.88, zorder=3)
    bars_cf   = ax.bar(x + w,   d_cf,   w, label='Control-Flow Restructuring', color=CF_COLOR, alpha=0.88, zorder=3)

    ax.axhline(0, color='black', linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('ΔF1 (percentage points)', fontsize=10)
    ax.set_title('Robustness of All Systems on Devign under Three Transformations', fontsize=10, fontweight='bold')

    # Family dividers
    dividers = [3.5, 5.5, 8.5]
    for d in dividers:
        ax.axvline(d, color='gray', linewidth=0.8, linestyle=':', alpha=0.7, zorder=2)

    # Family labels above the plot
    family_labels = [('Whole-fn GNN', 1.5), ('Graph+LM', 4.5), ('Transformer', 7.0), ('Non-neural', 9.5)]
    ymax = ax.get_ylim()[1]
    for label, xpos in family_labels:
        ax.text(xpos, ax.get_ylim()[1] * 0.97, label, ha='center', va='top',
                fontsize=7.5, style='italic', color='#444444',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0', edgecolor='none', alpha=0.7))

    ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax.set_ylim(-8.5, 8.5)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(OUT + 'fig4_delta_bar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig4_delta_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig4_delta_bar_chart saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Dead-Code ΔF1 Direction Reversal Across Datasets
# ════════════════════════════════════════════════════════════════════════════
def fig5_deadcode_reversal():
    # Selected representative systems (one per family + key outliers)
    # REVEAL excluded from BigVul (near-degenerate, floor effect)
    systems    = ['ECG RGCN', 'ANGLE', 'VulGNN', 'ReGVD', 'CodeBERT', 'CodeT5+', 'TF-IDF+LR']
    families   = ['W-GNN',    'W-GNN', 'W-GNN',  'G+LM',  'Trans',    'Trans',    'N-N']
    dead_dev   = [-0.14,  +1.38,  +1.24,  +1.12, -0.56, +0.64, +0.28]
    dead_big   = [-4.00,  -5.66,  -4.37, -28.95, -14.93, -17.23, -0.63]
    dead_div   = [+0.84,  +3.28,  +2.56,  +0.16,  -0.12, +0.47, +0.82]

    n = len(systems)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(x - w,   dead_dev, w, label='Devign',     color=DEV_COLOR,  alpha=0.88, zorder=3)
    ax.bar(x,       dead_div, w, label='DiverseVul', color=DIV_COLOR,  alpha=0.88, zorder=3)
    ax.bar(x + w,   dead_big, w, label='Big-Vul',    color=BIG_COLOR,  alpha=0.88, zorder=3)

    ax.axhline(0, color='black', linewidth=0.9, zorder=4)

    # Annotate the extreme BigVul bar for ReGVD
    idx_regvd = systems.index('ReGVD')
    ax.annotate('−28.95', xy=(idx_regvd + w, dead_big[idx_regvd]),
                xytext=(idx_regvd + w + 0.5, dead_big[idx_regvd] + 2.5),
                fontsize=7, color=BIG_COLOR,
                arrowprops=dict(arrowstyle='->', color=BIG_COLOR, lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel('ΔF1 under Dead-Code Insertion (pp)', fontsize=10)
    ax.set_title('Dead-Code Effect Reverses Across Datasets', fontsize=10, fontweight='bold')

    # Zero crossing annotation
    ax.text(n - 0.5, 0.8, '← positive on Devign & DiverseVul\n    negative on Big-Vul →',
            ha='right', va='bottom', fontsize=7, color='#555555', style='italic')

    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim(-33, 7)

    plt.tight_layout()
    plt.savefig(OUT + 'fig5_deadcode_reversal.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig5_deadcode_reversal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig5_deadcode_reversal saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — DeepWukong Pipeline Failure Diagram
# ════════════════════════════════════════════════════════════════════════════
def fig6_deepwukong_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        ('1\nParse\nSource', 1.0),
        ('2\nLocate\nSeed APIs', 3.0),
        ('3\nExtract\nXFG Slices', 5.0),
        ('4\nSymbolise\nIdentifiers', 7.0),
        ('5\nGNN\nClassifier', 9.0),
    ]

    # Failure annotations per stage
    failures = {
        1: ('Identifier\nRenaming', 'Seed names\nrenamed → not\nfound\n(XFGs: 7,893→3,935\nF1: 93.7→31.9%)', REN_COLOR),
        2: ('Dead-Code\nInsertion', 'Inflated XFGs\nwith noise\n(XFGs: 7,893→8,807\nF1: 93.7→66.2%)', DEAD_COLOR),
        4: ('Control-Flow\nRestructuring', 'Unfamiliar PDG\ntopology in GNN\n(XFGs: 7,893→5,511\nF1: 93.7→37.4%)', CF_COLOR),
    }

    box_w, box_h = 1.3, 0.85
    arrow_y = 2.2

    for i, (label, xc) in enumerate(stages):
        is_failure_stage = i in failures
        color = '#E3F2FD' if not is_failure_stage else '#FFEBEE'
        edge  = '#1565C0' if not is_failure_stage else '#C62828'
        lw    = 1.2 if not is_failure_stage else 2.0

        rect = FancyBboxPatch((xc - box_w/2, arrow_y - box_h/2), box_w, box_h,
                              boxstyle='round,pad=0.08', facecolor=color,
                              edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(rect)
        ax.text(xc, arrow_y, label, ha='center', va='center', fontsize=7.5,
                fontweight='bold', zorder=4, color='#1A1A2E')

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][1] - box_w/2, arrow_y),
                        xytext=(xc + box_w/2, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3),
                        zorder=2)

    # Failure annotations below each stage
    failure_xc_map = {0: 3.0, 1: 5.0, 4: 9.0}   # stage index → xc for annotation
    ann_y_base = 1.0

    for stage_idx, (transform_name, detail, color) in failures.items():
        xc = stages[stage_idx][1]
        # Vertical arrow pointing up to the box
        ax.annotate('', xy=(xc, arrow_y - box_h/2 - 0.02),
                    xytext=(xc, ann_y_base + 0.52),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=5)
        ax.text(xc, ann_y_base + 0.46, '[FAIL] ' + transform_name, ha='center', va='top',
                fontsize=7, color=color, fontweight='bold')
        ax.text(xc, ann_y_base + 0.05, detail, ha='center', va='top',
                fontsize=6.3, color='#333333',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8E1', edgecolor=color, alpha=0.85))

    # Legend patches
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label='Pipeline stage (unaffected)'),
        mpatches.Patch(facecolor='#FFEBEE', edgecolor='#C62828', label='Pipeline stage (failure point)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5, framealpha=0.9)

    ax.set_title('DeepWukong Pipeline: Stage-Specific Failure Under Each Transformation',
                 fontsize=10, fontweight='bold', pad=8)

    plt.tight_layout()
    plt.savefig(OUT + 'fig6_deepwukong_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig6_deepwukong_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig6_deepwukong_pipeline saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Budgeted Perturbation Attack (ΔF1 by budget level)
# ════════════════════════════════════════════════════════════════════════════
def fig7_budgeted_attack():
    """
    Data from 7-condition eval results.
    Budget 1 = worst single transform for each model.
    Budget 2 = worst pairwise combination.
    Budget 3 = compound (all three transforms).
    """
    # ReGVD (from regvd_7cond_results.json)
    regvd_b1 = min(-7.50, -8.85, -4.25)   # worst single = dead: -8.85
    regvd_b2 = min(-14.39, -14.79, -12.44) # worst pair = ren+cf: -14.79
    regvd_b3 = -19.40                       # compound

    # VulGNN (from vulgnn_7cond_results.json) — structural, immune
    vulgnn_b1 = min(0.00, 1.24, 0.61)     # worst single = ren: 0.00
    vulgnn_b2 = min(1.24, 0.61, 1.49)     # worst pair = ren+cf: 0.61
    vulgnn_b3 = 1.49                        # compound

    # ANGLE (from angle_7cond_results.json)
    angle_b1 = min(0.00, 1.38, -1.19)     # worst single = cf: -1.19
    angle_b2 = min(1.39, -1.19, 1.41)     # worst pair = ren+cf: -1.19
    angle_b3 = 1.41                         # compound

    # CodeBERT (from codebert_7cond_devign_results.json)
    codebert_b1 = min(-1.90, -1.71, -1.93) # worst single = cf: -1.93
    codebert_b2 = min(-1.94, -2.19, -1.85) # worst pair = ren+cf: -2.19
    codebert_b3 = -1.96                     # compound

    # TF-IDF (from tfidf_7cond_devign_results.json)
    tfidf_b1 = min(-4.75, 0.69, 0.44)     # worst single = ren: -4.75
    tfidf_b2 = min(-4.17, -3.12, 0.54)    # worst pair = ren+dead: -4.17
    tfidf_b3 = -3.20                        # compound

    budgets = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    models = {
        'ReGVD (Graph+LM)':         ([regvd_b1, regvd_b2, regvd_b3],    BIG_COLOR,   'o-',  2.2),
        'TF-IDF+LR (Non-neural)':   ([tfidf_b1, tfidf_b2, tfidf_b3],    '#6A1B9A',   's--', 1.5),
        'CodeBERT (Transformer)':   ([codebert_b1, codebert_b2, codebert_b3], DIV_COLOR, '^--', 1.5),
        'ANGLE (Structural GNN)':   ([angle_b1, angle_b2, angle_b3],     '#5C6BC0',   'D:',  1.5),
        'VulGNN (Structural GNN)':  ([vulgnn_b1, vulgnn_b2, vulgnn_b3],  DEV_COLOR,   'v:',  1.5),
    }

    for label, (vals, color, style, lw) in models.items():
        ax.plot(budgets, vals, style, label=label, color=color,
                linewidth=lw, markersize=7, zorder=3)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)
    ax.fill_between([0.8, 3.2], [-25, -25], [0, 0], color='#FFEBEE', alpha=0.25, zorder=1)
    ax.fill_between([0.8, 3.2], [0, 0],    [5, 5],  color='#E8F5E9', alpha=0.25, zorder=1)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Budget 1\n(1 transform)', 'Budget 2\n(2 transforms)', 'Budget 3\n(all 3)'],
                        fontsize=8.5)
    ax.set_ylabel('Worst-Case ΔF1 (percentage points)', fontsize=10)
    ax.set_title('Budgeted Perturbation Attack:\nDamage by Number of Transforms Applied', fontsize=10, fontweight='bold')
    ax.set_xlim(0.7, 3.3)
    ax.set_ylim(-24, 5)

    ax.text(3.05, 1.5, 'structural\nmodels\nimmune', fontsize=6.5, color='#2E7D32', va='center')
    ax.text(3.05, -10, 'token models\nscale with\nbudget', fontsize=6.5, color=BIG_COLOR, va='center')

    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT + 'fig7_budgeted_attack.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig7_budgeted_attack.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig7_budgeted_attack saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Clean F1 vs Renaming Robustness Scatter Plot
# ════════════════════════════════════════════════════════════════════════════
def fig8_scatter():
    # From Table 3 (clean F1 mean and ΔRen)
    data = [
        # (name, clean_f1, delta_ren, family)
        ('ECG RGCN',    63.48,  -0.43, 'W-GNN'),
        ('ANGLE',       59.98,   0.00, 'W-GNN'),
        ('VulGNN',      60.68,   0.00, 'W-GNN'),
        ('REVEAL',      59.45,  -6.08, 'W-GNN'),
        ('Vul-LMGGNN',  60.84,  -4.58, 'G+LM'),
        ('ReGVD',       62.01,  -1.95, 'G+LM'),
        ('CodeBERT',    63.54,  -1.45, 'Trans'),
        ('CodeT5+',     63.62,  -0.68, 'Trans'),
        ('CB-Aug',      61.44,  +0.08, 'Trans'),
        ('TF-IDF+LR',   57.93,  -4.02, 'N-N'),
        ('CPG+LR',      54.67,  -6.30, 'N-N'),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    plotted_families = set()
    for name, f1, dr, fam in data:
        color = FAM_COLORS[fam]
        label = {'W-GNN': 'Whole-fn GNN', 'G+LM': 'Graph+LM',
                 'Trans': 'Transformer', 'N-N': 'Non-neural'}[fam]
        lbl = label if fam not in plotted_families else '_nolegend_'
        plotted_families.add(fam)
        ax.scatter(f1, dr, color=color, s=75, label=lbl, zorder=4, edgecolors='white', linewidths=0.6)

        offset = (1.5, 0.1)
        if name == 'REVEAL': offset = (1.0, -0.5)
        if name == 'CPG+LR': offset = (-4.5, 0.2)
        if name == 'TF-IDF+LR': offset = (-7.0, -0.3)
        if name == 'ANGLE': offset = (0.5, 0.3)
        if name == 'VulGNN': offset = (0.5, -0.5)
        ax.annotate(name, (f1, dr), xytext=(f1 + offset[0], dr + offset[1]),
                    fontsize=6.8, color='#333333', zorder=5)

    # Fit line (just visual, no real fit shown - just horizontal bands)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # Shaded regions
    ax.fill_between([52, 66], [-7.5, -7.5], [-2.5, -2.5],
                    color='#FFEBEE', alpha=0.4, zorder=1, label='_nolegend_')
    ax.fill_between([52, 66], [-1.0, -1.0], [1.0, 1.0],
                    color='#E8F5E9', alpha=0.4, zorder=1, label='_nolegend_')

    ax.set_xlabel('Clean F1 (%)', fontsize=10)
    ax.set_ylabel('ΔF1 under Identifier Renaming (pp)', fontsize=10)
    ax.set_title('Clean Accuracy Does Not Predict\nRobustness to Identifier Renaming', fontsize=10, fontweight='bold')
    ax.set_xlim(52, 66)
    ax.set_ylim(-7.5, 2.0)

    ax.text(64.5, 0.5, 'robust\nzone', fontsize=7, color='#2E7D32', ha='center')
    ax.text(64.5, -5.5, 'fragile\nzone', fontsize=7, color='#C62828', ha='center')

    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT + 'fig8_scatter_f1_vs_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig8_scatter_f1_vs_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig8_scatter_f1_vs_robustness saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Distribution Shift Quantification
# ════════════════════════════════════════════════════════════════════════════
def fig9_distribution_shift():
    # From token_dist_analysis.json (actual values)
    conditions = ['Original', 'Identifier\nRenaming', 'Dead-Code\nInsertion', 'Control-Flow\nRestructuring']
    tfidf_sim  = [0.991, 0.873, 0.917, 0.962]
    bert_sim   = [1.000, 0.998, 0.992, 0.998]   # original is reference = 1.0
    oov_rate   = [3.68, 4.26, 3.64, 3.73]

    x = np.arange(len(conditions))
    colors = ['#78909C', REN_COLOR, DEAD_COLOR, CF_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Panel 1: TF-IDF centroid cosine similarity
    bars1 = ax1.bar(x, tfidf_sim, color=colors, alpha=0.88, zorder=3, width=0.55, edgecolor='white')
    ax1.set_ylim(0.82, 1.01)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=8)
    ax1.set_ylabel('TF-IDF Centroid Cosine Similarity', fontsize=9)
    ax1.set_title('Vocabulary-Level Shift', fontsize=9, fontweight='bold')
    ax1.axhline(1.0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    for bar, val in zip(bars1, tfidf_sim):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Panel 2: CodeBERT [CLS] similarity + OOV rate (twin axis)
    bars2 = ax2.bar(x, bert_sim, color=colors, alpha=0.88, zorder=3, width=0.55, edgecolor='white')
    ax2.set_ylim(0.985, 1.003)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=8)
    ax2.set_ylabel('CodeBERT [CLS] Pairwise Cosine Similarity', fontsize=9)
    ax2.set_title('Semantic-Level Shift', fontsize=9, fontweight='bold')
    ax2.axhline(1.0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    for bar, val in zip(bars2, bert_sim):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax2.text(1.5, 0.9865, 'Note: Semantic similarity stays >= 0.99\ndespite large vocabulary shift',
             fontsize=7, color='#2E7D32', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8))

    fig.suptitle('Transformations Alter Token Distribution But Preserve Meaning',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig9_distribution_shift.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig9_distribution_shift.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig9_distribution_shift saved")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Evaluation Pipeline Architecture
# ════════════════════════════════════════════════════════════════════════════
def fig10_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(ax, x, y, w, h, text, fc, ec, fontsize=8, bold=False, text_color='#1A1A2E'):
        r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle='round,pad=0.12', facecolor=fc, edgecolor=ec,
                           linewidth=1.5, zorder=3)
        ax.add_patch(r)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold' if bold else 'normal', color=text_color,
                zorder=4, multialignment='center')

    def arrow(ax, x1, y1, x2, y2, color='#555555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.4), zorder=5)

    def label(ax, x, y, text, fs=7.5, color='#444444'):
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color=color, style='italic', zorder=6)

    # ── Row 1: Datasets ──────────────────────────────────────────────────
    ax.text(6, 6.6, 'EVALUATION PIPELINE ARCHITECTURE', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1A237E')

    dset_y = 5.9
    datasets = [('Devign\n27,318 fns', 2.2), ('Big-Vul\n188,636 fns', 6.0), ('DiverseVul\n36,598 fns', 9.8)]
    for (txt, xc) in datasets:
        box(ax, xc, dset_y, 2.5, 0.65, txt, '#E3F2FD', '#1565C0', fontsize=8, bold=True)

    # DeepWukong separate
    box(ax, 11.5, dset_y, 1.7, 0.65, 'SARD\nCWE-119', '#F3E5F5', '#6A1B9A', fontsize=8, bold=True)

    # ── Row 2: Transformation layer ──────────────────────────────────────
    transform_y = 4.85
    ax.text(5.2, transform_y + 0.58, 'Code Transformations (test set only)', ha='center',
            fontsize=8, color='#BF360C', fontweight='bold')

    transforms = [
        ('Original\n(baseline)', '#ECEFF1', '#607D8B', 1.0),
        ('Identifier\nRenaming',  '#FFEBEE', REN_COLOR,  2.8),
        ('Dead-Code\nInsertion',  '#E3F2FD', DEAD_COLOR, 4.6),
        ('Control-Flow\nRestructuring', '#E8F5E9', CF_COLOR, 6.4),
        ('Pairwise\n(×3 pairs)',  '#FFF3E0', '#E65100', 8.2),
        ('Compound\n(all three)', '#FCE4EC', '#880E4F', 10.0),
    ]
    for (txt, fc, ec, xc) in transforms:
        box(ax, xc, transform_y, 1.55, 0.65, txt, fc, ec, fontsize=7.2)

    # Note: pairwise + compound only for Devign
    ax.text(9.1, transform_y - 0.55, '← Devign only →', ha='center', fontsize=7,
            color='#E65100', style='italic')

    # Arrows from datasets to transforms
    for xc in [2.2, 6.0, 9.8]:
        arrow(ax, xc, dset_y - 0.33, 5.5, transform_y + 0.33, '#555555')

    # ── Row 3: Feature extraction ─────────────────────────────────────────
    feat_y = 3.65
    ax.text(3.5, feat_y + 0.58, 'Feature Extraction', ha='center', fontsize=8,
            color='#333333', fontweight='bold')

    box(ax, 2.2, feat_y, 2.6, 0.65,
        'CPG via Joern\n(nodes: stmt type, data/ctrl edges)',
        '#E8EAF6', '#3949AB', fontsize=7.5)
    box(ax, 5.2, feat_y, 2.6, 0.65,
        'Token Sequences\n(sub-word BPE tokenization)',
        '#E0F2F1', '#00695C', fontsize=7.5)
    box(ax, 8.2, feat_y, 2.6, 0.65,
        'TF-IDF\n(bag-of-tokens unigrams)',
        '#F3E5F5', '#6A1B9A', fontsize=7.5)

    arrow(ax, 5.5, transform_y - 0.33, 2.2, feat_y + 0.33)
    arrow(ax, 5.5, transform_y - 0.33, 5.2, feat_y + 0.33)
    arrow(ax, 5.5, transform_y - 0.33, 8.2, feat_y + 0.33)

    # ── Row 4: Models ─────────────────────────────────────────────────────
    model_y = 2.4
    ax.text(5.2, model_y + 0.62, 'Models (5 random seeds each)', ha='center', fontsize=8,
            color='#333333', fontweight='bold')

    model_groups = [
        ('Whole-fn GNN\nECG RGCN · ANGLE\nVulGNN · REVEAL', 1.5, '#BBDEFB', '#1565C0'),
        ('Graph + LM\nReGVD\nVul-LMGGNN',                    3.9, '#FFCCBC', '#BF360C'),
        ('Transformer\nCodeBERT · CodeT5+\nCodeBERT-Aug',     6.3, '#C8E6C9', '#2E7D32'),
        ('Non-neural\nTF-IDF+LR\nCPG+LR',                    8.7, '#E1BEE7', '#6A1B9A'),
    ]
    for (txt, xc, fc, ec) in model_groups:
        box(ax, xc, model_y, 2.1, 0.85, txt, fc, ec, fontsize=7.2)

    # Arrows from feature extraction to models
    arrow(ax, 2.2, feat_y - 0.33, 1.5, model_y + 0.43)   # CPG → W-GNN
    arrow(ax, 2.2, feat_y - 0.33, 3.9, model_y + 0.43)   # CPG → G+LM
    arrow(ax, 5.2, feat_y - 0.33, 3.9, model_y + 0.43)   # tokens → G+LM
    arrow(ax, 5.2, feat_y - 0.33, 6.3, model_y + 0.43)   # tokens → Transformer
    arrow(ax, 8.2, feat_y - 0.33, 8.7, model_y + 0.43)   # TF-IDF → Non-neural

    # ── Row 5: Metrics ────────────────────────────────────────────────────
    metric_y = 1.2
    box(ax, 5.2, metric_y, 8.5, 0.65,
        'Evaluation   ·   F1  ·  ΔF1 per condition  ·  Attack Success Rate (ASR)  ·  5-seed mean ± std  ·  Bonferroni-corrected t-tests',
        '#FFFDE7', '#F57F17', fontsize=8, bold=False)

    for xc in [1.5, 3.9, 6.3, 8.7]:
        arrow(ax, xc, model_y - 0.43, 5.2, metric_y + 0.33)

    # DeepWukong separate path on the right
    box(ax, 11.5, 4.85, 1.55, 0.65, 'Pipeline\n5 Stages', '#F3E5F5', '#6A1B9A', fontsize=7.5)
    box(ax, 11.5, 3.65, 1.55, 0.65, 'XFG\nExtraction', '#F3E5F5', '#6A1B9A', fontsize=7.5)
    box(ax, 11.5, 2.40, 1.55, 0.85, 'DeepWukong\nGNN', '#E1BEE7', '#6A1B9A', fontsize=7.5)
    arrow(ax, 11.5, dset_y - 0.33, 11.5, 4.85 + 0.33)
    arrow(ax, 11.5, 4.85 - 0.33,   11.5, 3.65 + 0.33)
    arrow(ax, 11.5, 3.65 - 0.33,   11.5, 2.40 + 0.43)
    arrow(ax, 11.5, 2.40 - 0.43,   9.45, metric_y + 0.33)
    label(ax, 11.5, 0.75, 'SARD CWE-119\n(separate evaluation)', fs=7, color='#6A1B9A')

    plt.tight_layout(pad=0.5)
    plt.savefig(OUT + 'fig10_pipeline_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig10_pipeline_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig10_pipeline_architecture saved")


# ════════════════════════════════════════════════════════════════════════════
# fig_dwk_breakdown — XFG count (bars) + F1 (line) for DeepWukong on SARD
# ════════════════════════════════════════════════════════════════════════════
def fig_dwk_breakdown():
    conditions = ['Clean', 'Rename', 'Dead', 'CF', 'R+D', 'R+CF', 'D+CF', 'Cmp']
    xfg_counts = [7893,      3935,     8807,  5511,  4849,  3762,   6425,  4676]
    f1_scores  = [93.7,      31.9,     66.2,  37.4,  14.3,   9.6,   15.4,   6.1]

    x = np.arange(len(conditions))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))

    bars = ax1.bar(x, xfg_counts, color='#90CAF9', width=0.55,
                   edgecolor='#5A9FD4', linewidth=0.6, zorder=2, label='XFG count')
    ax1.set_ylabel('XFG slices extracted', fontsize=12.5)
    ax1.set_ylim(0, max(xfg_counts) * 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=12.5)
    ax1.tick_params(axis='y', labelsize=12.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    ax2 = ax1.twinx()
    line, = ax2.plot(x, f1_scores, color='#8B0000', linewidth=2.2,
                     marker='o', markersize=6, zorder=3, label='F1 (%)')
    for xi, fi in zip(x, f1_scores):
        offset = 3.5 if fi < 85 else -8
        va = 'bottom' if fi < 85 else 'top'
        ax2.text(xi, fi + offset, f'{fi:.1f}', ha='center', va=va,
                 fontsize=12.5, color='#8B0000')
    ax2.set_ylabel('F1 score (%)', fontsize=12.5, color='#8B0000')
    ax2.tick_params(axis='y', labelsize=12.5, colors='#8B0000')
    ax2.set_ylim(0, 115)
    ax2.spines['right'].set_color('#8B0000')

    ax1.set_title('DeepWukong on SARD CWE-119: XFG count and F1 per condition',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
    ax1.set_axisbelow(True)

    handles = [bars, line]
    labels  = ['XFG count', 'F1 (%)']
    ax1.legend(handles, labels, fontsize=12.5, loc='upper right', framealpha=0.85)

    plt.tight_layout(pad=0.8)
    plt.savefig(OUT + 'fig_dwk_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig_dwk_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig_dwk_breakdown saved")


# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import matplotlib.ticker
    print("Generating figures...")
    fig4_delta_bar_chart()
    fig5_deadcode_reversal()
    fig6_deepwukong_pipeline()
    fig7_budgeted_attack()
    fig8_scatter()
    fig9_distribution_shift()
    fig10_pipeline_architecture()
    fig_dwk_breakdown()
    print(f"\nAll figures saved to {OUT}")
