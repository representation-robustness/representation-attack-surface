"""Drop-in LaTeX for paper Tables III-X from the RERUN_ROOT results (read-only on results; writes OUT only).
Run: RERUN_ROOT=<rerun root> PYTHONPATH=<repo> <venv python> make_paper_tables.py OUT_DIR"""
import os
import csv, glob, gzip, json, sys
from pathlib import Path
import numpy as np

R = Path(os.environ.get("RERUN_ROOT", "../rerun")) / "results"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["identifier", "deadcode", "controlflow", "ren_dead", "ren_cf", "dead_cf", "compound"]
HDR = ("Family & System & Clean F1 & $\\Delta$Ren & $\\Delta$Dead & $\\Delta$CF &\n"
       "  $\\Delta$R+D & $\\Delta$R+CF & $\\Delta$D+CF & $\\Delta$Cmp & Max~$|\\Delta|$ \\\\")


def num(x, nd=2, sign=True, bold=False):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
    elif sign and x > 0:
        s = "+" + s
    return f"$\\mathbf{{{s}}}$" if bold else (f"${s}$" if sign and float(s) != 0 else s)


def pm(m, s, nd=2):
    return f"{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}"


def stats(ds):
    d = {}
    for r in json.load(open(R / f"stats/{ds}_valid.json")):
        d.setdefault(r["model"], {})[r["condition"]] = r
    return d


def robustness(ds, groups, caption, label, placement, fname):
    st = stats(ds)
    lines = [f"\\begin{{table*}}{placement}", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llrrrrrrrrr}",
             "\\toprule", HDR, "\\midrule"]
    for gi, (fam, rows) in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        for ri, (model, name) in enumerate(rows):
            s = st[model]
            c0 = s["identifier"]
            cells = [num(s[c]["delta_f1"], bold=bool(s[c]["significant"])) for c in CONDS]
            mx = max(abs(s[c]["delta_f1"]) for c in CONDS)
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{fam}}}\n" if ri == 0 else ""
            lines.append(f"{first} & {name} & {pm(c0['clean_f1_full_test'], c0['clean_f1_full_test_sd'])} & "
                         + " & ".join(cells) + f" & {mx:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\vspace{-2mm}", "\\end{table*}"]
    (OUT / fname).write_text("\n".join(lines) + "\n")


SIG = ("\\textbf{bold} = statistically significant (neural: paired $t$-test over five seeds, BH FDR $q<0.05$ "
       "within the table; non-neural: 95\\% interval over 30 training subsamples excludes 0)")
VALID = "$\\Delta$ values are F1 changes relative to clean performance on the same functions, restricted to valid transformations"
W = [("ecg_rgcn", "ECG RGCN"), ("angle", "ANGLE"), ("vulgnn", "VulGNN"), ("reveal", "REVEAL")]
GLM = ("G+LM", [("lmggnn", "Vul-LMGGNN"), ("regvd", "ReGVD")])
NN = ("N-N", [("tfidf_lr", "TF-IDF$+$LR"), ("cpg_lr", "CPG$+$LR")])
TR = [("codebert", "CodeBERT$^*$"), ("codet5plus", "CodeT5+$^*$")]

robustness("devign", [("W-GNN", W + [("devign_ggnn", "Dev.\\ GGNN$\\dagger$")]), GLM,
                      ("Trans", TR + [("codebert_aug", "CodeBERT-Aug$^*$$\\ddagger$")]), NN],
           f"Devign robustness results. Clean F1: mean\\,$\\pm$\\,std. {VALID}; {SIG}. "
           "$\\dagger$~near-degenerate (flags 79--95\\% of test functions); $*$~sequence input; $\\ddagger$~augmented training.",
           "tab:devign", "", "table3_devign.tex")
robustness("diversevul", [("W-GNN", W), GLM, ("Trans", TR), NN],
           f"DiverseVul robustness results. Clean F1: mean\\,$\\pm$\\,std. {VALID}; {SIG}. $*$~sequence input.",
           "tab:diversevul", "[t]", "table4_diversevul.tex")
robustness("bigvul", [("W-GNN", W), GLM, ("Trans", TR), NN],
           "Big-Vul robustness results under seven conditions. Clean F1 is reported\nas mean $\\pm$ std. "
           f"{VALID};\n{SIG}.\n$*$ is the sequence input.",
           "tab:bigvul", "[htbp]", "table5_bigvul.tex")

# ---- Table VI: identifier-exposure ablation
ab = list(csv.DictReader(open(R / "tables/table_ablation_devign.csv")))
NAMES = {"reveal": "REVEAL", "ecg_rgcn": "ECG RGCN", "angle": "ANGLE", "vulgnn": "VulGNN"}
VAR = {"reveal": "Full (baseline)", "ecg_rgcn": "Full (baseline)", "angle": "Full (baseline)", "vulgnn": "Full (baseline)",
       "reveal_struct": "struct-only", "ecg_rgcn_struct": "struct-only", "angle_struct": "struct-only",
       "vulgnn_struct": "struct-only", "vulgnn_struct_cb": "struct-only\\,+CodeBERT"}
L = ["\\begin{table}[h]", "\\centering", "\\small",
     "\\caption{Identifier-exposure ablation on Devign. Clean F1 and Ren F1 are reported as mean $\\pm$ std across five seeds,\n"
     "on the functions whose renaming is valid. struct-only keeps only the node-type encoding; +CodeBERT adds a\n"
     "function-level CodeBERT [CLS] embedding to struct-only VulGNN.}",
     "\\label{tab:causal_ablation}", "\\setlength{\\tabcolsep}{2.5pt}", "\\renewcommand{\\arraystretch}{0.95}",
     "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llrrr}", "\\toprule",
     "Model & Variant & Clean F1 & Ren F1 & $\\Delta$Ren \\\\", "\\midrule"]
prev = None
seen = set()
for r in ab:
    if (r["base"], r["variant"]) in seen or r["variant"] == "vulgnn_struct" and r["base"] == "vulgnn_struct":
        continue
    seen.add((r["base"], r["variant"]))
    arch = r["base"].replace("_struct", "")
    if prev is not None and arch != prev:
        L.append("\\midrule")
    prev = arch
    d = float(r["delta_ren"])
    s = f"{d:+.2f}" if round(d, 2) != 0 else "\\phantom{+}0.00"
    L.append(f"{NAMES[arch]} & {VAR[r['variant']]} & ${float(r['clean_f1']):.2f} \\pm {float(r['clean_sd']):.2f}$ & "
             f"${float(r['ren_f1']):.2f} \\pm {float(r['ren_sd']):.2f}$ & ${s}$ \\\\")
L += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
(OUT / "table6_ablation.tex").write_text("\n".join(L) + "\n")

# ---- Table VII: minimum budget
mb = {r["model"]: r for r in csv.DictReader(open(R / "tables/table_minbudget_devign.csv"))}
ROWS7 = [("ecg_rgcn", "ECG RGCN"), ("angle", "ANGLE"), ("vulgnn", "VulGNN"), ("reveal", "REVEAL"),
         ("lmggnn", "Vul-LMGGNN"), ("regvd", "ReGVD"), ("codebert", "CodeBERT"), ("codet5plus", "CodeT5+"),
         ("tfidf_lr", "TF-IDF$+$LR"), ("cpg_lr", "CPG$+$LR"), None,
         ("ecg_rgcn_struct", "ECG RGCN-S$^\\dagger$"), ("angle_struct", "ANGLE-S$^\\dagger$"),
         ("vulgnn_struct", "VulGNN-S$^\\dagger$"), ("reveal_struct", "REVEAL-S$^\\dagger$")]
L = ["\\begin{table}[htbp]", "\\centering", "\\small",
     "\\caption{Minimum-budget attack on Devign. ASR@$k$ is the\ncumulative fraction of correctly classified vulnerable "
     "functions evaded at budget\n$k$ using only valid transformations (mean\\,$\\pm$\\,std over runs). $\\dagger$ denotes "
     "renaming-invariant (node-type-only) models.}",
     "\\label{tab:greedy}", "\\setlength{\\tabcolsep}{4pt}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrr}",
     "  \\toprule", "  System & ASR@1 & ASR@2 & ASR@3 & Mean budget \\\\", "  \\midrule"]
for x in ROWS7:
    if x is None:
        L.append("  \\midrule"); continue
    r = mb[x[0]]
    L.append(f"  {x[1]} & " + " & ".join(pm(float(r[f'asr{k}']), float(r[f'asr{k}_sd']), 1) for k in (1, 2, 3))
             + f" & {float(r['mean_budget']):.2f} \\\\")
L += ["  \\bottomrule", "\\end{tabular}", "}", "\\vspace{-2mm}", "\\end{table}"]
(OUT / "table7_minbudget.tex").write_text("\n".join(L) + "\n")

# ---- Table VIII: transfer
tr = {(r["source"], r["target"]): float(r["tasr_mean"]) for r in csv.DictReader(open(R / "tables/table_transfer_devign.csv"))}
M8 = [("angle", "ANGLE"), ("vulgnn", "VulGNN"), ("ecg_rgcn", "ECG RGCN"), ("reveal", "REVEAL"),
      ("regvd", "ReGVD"), ("lmggnn", "Vul-LMGGNN"), ("codebert", "CodeBERT"), ("codet5plus", "CodeT5+")]
COLS = {"ecg_rgcn": "ECG"}
L = ["\\begin{table}[t]", "\\centering", "\\small",
     "\\caption{Transfer attack success rate (TASR, \\%) on Devign. Among functions both detectors classify correctly, "
     "the cell reports\nthe fraction of source-evaded functions that are also evaded at the target\n"
     "using the source model's minimum-budget transformation (mean over seed-paired runs).}",
     "\\label{tab:transfer_asr}", "\\setlength{\\tabcolsep}{2pt}", "\\renewcommand{\\arraystretch}{1.0}",
     "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lcccccccc}", "\\toprule",
     "Source $\\rightarrow$ Target\n  & \\multicolumn{4}{c}{W-GNN}\n  & \\multicolumn{2}{c}{Graph+LM}\n"
     "  & \\multicolumn{2}{c}{Transformer} \\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
     " & " + " & ".join(COLS.get(m, n) for m, n in M8) + " \\\\", "\\midrule"]
for i, (s, sn) in enumerate(M8):
    if i in (4, 6):
        L.append("\\midrule")
    L.append(f"{sn:<12} & " + " & ".join("--" if s == t else f"{tr[(s, t)]:5.2f}" for t, _ in M8) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "}", "%\\vspace{-4mm}", "\\end{table}"]
(OUT / "table8_transfer.tex").write_text("\n".join(L) + "\n")

# ---- Table IX: mitigation
mi = {(r["model"], r["variant"]): r for r in csv.DictReader(open(R / "tables/table_mitigation_devign.csv"))}
L = ["\\begin{table}[t]", "\\centering", "\\small",
     "\\caption{Robustness training across detector families on Devign. Augmented models\nare trained on clean functions "
     "plus transformed variants (Ren, Dead, CF); Vol.\\ models are trained on the\nclean functions repeated to the same "
     "training-set size (volume control).}",
     "\\label{tab:mitigation_all}", "\\setlength{\\tabcolsep}{2.5pt}", "\\renewcommand{\\arraystretch}{0.95}",
     "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llrrrrr}", "\\toprule",
     "Family & Model & Clean F1 & $\\Delta$Ren & $\\Delta$Dead & $\\Delta$CF & Max~$|\\Delta|$ \\\\", "\\midrule"]
for fam, m, n in [("Transformer", "codebert", "CodeBERT"), ("G+LM", "regvd", "ReGVD"), ("W-GNN", "reveal", "REVEAL")]:
    for v, suf in [(m, ""), (f"{m}_volctl", "-Vol"), (f"{m}_aug", "-Aug")]:
        r = mi[(m, v)]
        L.append(f"{fam:<11} & {n + suf:<14} & {float(r['clean_f1']):.2f} & "
                 + " & ".join(num(float(r[f"delta_{c}"])) for c in ("identifier", "deadcode", "controlflow"))
                 + f" & {float(r['max_abs_delta']):.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "}", "%\\vspace{-2mm}", "\\end{table}"]
(OUT / "table9_mitigation.tex").write_text("\n".join(L) + "\n")

# ---- Table X: DeepWukong (paired: authors' test XFGs whose slice survives the transformation)
from pipeline.deepwukong.report import _m, load_index  # noqa: E402
runs = [json.load(gzip.open(f)) for f in sorted(glob.glob(str(R / "preds/deepwukong/deepwukong/run_*.json.gz")))]
dst = json.load(open(R / "stats/deepwukong.json"))
clean_idx = load_index("clean")
cf1 = [x["f1"] for x in dst["clean_shipped"]]
NM = {"identifier": "Identifier renaming", "deadcode": "Dead-code insertion", "controlflow": "Control-flow restructuring",
      "ren_dead": "Ren + Dead", "ren_cf": "Ren + CF", "dead_cf": "Dead + CF", "compound": "Compound (Ren+Dead+CF)"}
L = ["\\begin{table}[tbp]", "\\centering",
     "\\caption{DeepWukong under clean and transformed conditions (SARD CWE-119, authors' test split, five seeds).\n"
     "XFGs: authors' test slices whose seed statement and slice are still produced after the transformation;\n"
     "$\\Delta$F1 is measured relative to the clean condition on the same slices.}",
     "\\label{tab:deepwukong}", "\\begin{tabular}{lrrr}", "  \\toprule",
     "  Condition & XFGs & F1 (\\%) & $\\Delta$F1 (pp) \\\\", "  \\midrule",
     f"  Original (clean)            & {dst['clean_shipped'][0]['n']:,} & {pm(np.mean(cf1), np.std(cf1, ddof=1))} & --- \\\\"]
for c in CONDS:
    idx = load_index(c)
    scope = {i for i, x in idx.items() if x["found"] and x["valid_scope"] and clean_idx.get(i, {}).get("found")}
    f1s = []
    for r in runs:
        t, cl, trr = r["threshold"], r["test"]["clean"], r["test"][c]
        pt = dict(zip(trr["ids"], trr["p"])); pc = dict(zip(cl["ids"], cl["p"])); yc = dict(zip(cl["ids"], cl["y"]))
        ids = sorted(scope & pc.keys() & pt.keys())
        f1s.append(_m([yc[i] for i in ids], [pt[i] for i in ids], t)["f1"])
    p = dst["conditions"][c]["paired"]
    if c == "ren_dead":
        L.append("  \\midrule")
    L.append(f"  {NM[c]:<27} & {p['n']:,} & {pm(np.mean(f1s), np.std(f1s, ddof=1))} & ${p['f1']['delta_mean']:+.2f}$ \\\\")
L += ["  \\bottomrule", "\\end{tabular}", "\\vspace{-2mm}", "\\end{table}"]
(OUT / "table10_deepwukong.tex").write_text("\n".join(L) + "\n")
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
