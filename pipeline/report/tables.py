"""Generate every paper table from result files (no hand-copied numbers).

  python -m pipeline.report.tables            # writes results/tables/*.tex and *.csv

  table_validity.tex           transformation change / parse / compile validity per dataset x condition
  table_robustness_<ds>.tex    clean F1 (mean +- sd) and Delta F1 per condition on validly transformed
                               samples; bold = significant (BH q<0.05 for neural, 95% interval excludes 0
                               for non-neural); run counts in a column
  table_hparams.tex            Table I, from configs/*.yaml
"""
import json

import yaml

from pipeline.models.common import CONDITIONS, REPO, ROOT

OUT = ROOT / "results" / "tables"
DATASETS = {"devign": "Devign", "diversevul": "DiverseVul", "bigvul": "Big-Vul"}
COND_HDR = ["$\\Delta$Ren", "$\\Delta$Dead", "$\\Delta$CF", "$\\Delta$R+D", "$\\Delta$R+CF", "$\\Delta$D+CF", "$\\Delta$Cmp"]
COND_SHORT = ["Ren", "Dead", "CF", "R+D", "R+CF", "D+CF", "Cmp"]
# Main rows are each detector as its paper specifies its inputs; the "-S" rows are the same architectures
# fed node types only, a controlled ablation of the representation assumption (not separate systems).
FAMILY = [
    ("Whole-fn GNN", ["ecg_rgcn", "angle", "vulgnn", "reveal", "devign_ggnn"]),
    ("Graph + LM", ["lmggnn", "regvd"]),
    ("Transformer", ["codebert", "codet5plus"]),
    ("Non-neural", ["tfidf_lr", "cpg_lr"]),
    ("Ablation: node types only", ["ecg_rgcn_struct", "angle_struct", "vulgnn_struct", "reveal_struct", "vulgnn_struct_cb"]),
]
NAMES = {"ecg_rgcn": "ECG RGCN", "ecg_rgcn_struct": "ECG RGCN-S", "angle": "ANGLE", "angle_struct": "ANGLE-S",
         "vulgnn": "VulGNN", "vulgnn_struct": "VulGNN-S", "reveal": "REVEAL", "reveal_struct": "REVEAL-S",
         "devign_ggnn": "Devign GGNN", "vulgnn_struct_cb": "VulGNN-S + CodeBERT", "lmggnn": "Vul-LMGGNN", "regvd": "ReGVD", "codebert": "CodeBERT",
         "codet5plus": "CodeT5+", "tfidf_lr": "TF-IDF+LR", "cpg_lr": "CPG+LR"}


def validity_table():
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             "Dataset & Condition & $n$ & Func.\\% & Changed\\% & Valid\\% & Valid$\\mid$Chg.\\% \\\\", "\\midrule"]
    csv = ["dataset,condition,n,func_found,changed,valid,valid_given_changed"]
    for ds, nice in DATASETS.items():
        s = json.loads((ROOT / "data" / ds / "validity" / "test" / "summary.json").read_text())["conditions"]
        for i, (c, short) in enumerate(zip(CONDITIONS, COND_SHORT)):
            v = s[c]
            vals = [100 * v["rate_func_found"], 100 * v["rate_changed"], 100 * v["rate_valid"],
                    100 * v["rate_valid_given_changed"]]
            lines.append(f"{nice if i == 0 else ''} & {short} & {v['n']} & " + " & ".join(f"{x:.1f}" for x in vals) + " \\\\")
            csv.append(f"{ds},{c},{v['n']}," + ",".join(f"{x:.2f}" for x in vals))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (OUT / "table_validity.tex").write_text("\n".join(lines) + "\n")
    (OUT / "table_validity.csv").write_text("\n".join(csv) + "\n")


def robustness_table(ds: str):
    path = ROOT / "results" / "stats" / f"{ds}_valid.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    by = {(r["model"], r["condition"]): r for r in rows}
    lines = ["\\begin{tabular}{llrr" + "r" * len(CONDITIONS) + "r}", "\\toprule",
             "Family & System & Runs & Clean F1 & " + " & ".join(COND_HDR) + " & Max$|\\Delta|$ \\\\", "\\midrule"]
    for fam, models in FAMILY:
        present = [m for m in models if (m, CONDITIONS[0]) in by]
        for i, m in enumerate(present):
            first = by[(m, CONDITIONS[0])]
            cells, mx = [], 0.0
            for c in CONDITIONS:
                r = by.get((m, c))
                if r is None:
                    cells.append("--")
                    continue
                d = r["delta_f1"]
                mx = max(mx, abs(d))
                txt = f"{d:+.2f}"
                cells.append(f"$\\mathbf{{{txt}}}$" if r.get("significant") else f"${txt}$")
            lines.append(f"{fam if i == 0 else ''} & {NAMES.get(m, m)} & {first['n_runs']} & "
                         f"{first['clean_f1_full_test']:.2f}\\,$\\pm$\\,{first['clean_f1_full_test_sd']:.2f} & " + " & ".join(cells) + f" & {mx:.2f} \\\\")
        if present:
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (OUT / f"table_robustness_{ds}.tex").write_text("\n".join(lines) + "\n")


def hparams_table():
    m = yaml.safe_load(open(REPO / "configs/models.yaml"))
    g = yaml.safe_load(open(REPO / "configs/graph_models.yaml"))
    lines = ["\\begin{tabular}{lrrrl}", "\\toprule", "System & Batch & LR & Epochs (ES) & Node features \\\\", "\\midrule"]
    for k, c in g.items():
        base = dict(c)
        while "inherit" in base:
            parent = dict(g[base.pop("inherit")])
            parent.update(base)
            base = parent
        lines.append(f"{NAMES.get(k, k)} & {base['batch_size']} & {base['lr']:g} & {base['epochs']} ({base['patience']}) & "
                     f"{base['view']['features']} \\\\")
    for k in ("codebert", "codet5plus"):
        c = m[k]
        lines.append(f"{NAMES[k]} & {c['batch_size']} & {c['lr']:g} & {c['epochs']} (--) & tokens (max {c['max_len']}) \\\\")
    lines += ["ReGVD & 128 & 5e-4 & 100 (10) & GraphCodeBERT token graph \\\\",
              "Vul-LMGGNN & 8 & 2e-5 & 10 (--) & type + Word2Vec, CodeBERT \\\\",
              "TF-IDF+LR / CPG+LR & -- & -- & $C=1$ & n-grams / type histograms \\\\", "\\bottomrule", "\\end{tabular}"]
    (OUT / "table_hparams.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validity_table()
    for ds in DATASETS:
        robustness_table(ds)
    hparams_table()
    print("tables ->", OUT, sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
