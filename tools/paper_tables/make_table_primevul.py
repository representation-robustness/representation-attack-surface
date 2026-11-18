"""PrimeVul robustness table (same format as Tables III-V), from the PrimeVul stats.
Reads PRIMEVUL_STATS (default $RERUN_ROOT/results/stats/primevul_valid.json); writes table_primevul.tex here."""
import json, os
from pathlib import Path

SRC = Path(os.environ.get("PRIMEVUL_STATS", Path(os.environ.get("RERUN_ROOT", "../rerun")) / "results/stats/primevul_valid.json"))
OUT = Path(__file__).resolve().parent
CONDS = ["identifier", "deadcode", "controlflow", "ren_dead", "ren_cf", "dead_cf", "compound"]
HDR = ("Family & System & Clean F1 & $\\Delta$Ren & $\\Delta$Dead & $\\Delta$CF &\n"
       "  $\\Delta$R+D & $\\Delta$R+CF & $\\Delta$D+CF & $\\Delta$Cmp & Max~$|\\Delta|$ \\\\")


def num(x, nd=2, bold=False):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
        return f"$\\mathbf{{{s}}}$" if bold else s
    if x > 0:
        s = "+" + s
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


st = {}
for r in json.load(open(SRC)):
    st.setdefault(r["model"], {})[r["condition"]] = r
W = [("ecg_rgcn", "ECG RGCN"), ("angle", "ANGLE"), ("vulgnn", "VulGNN"), ("reveal", "REVEAL")]
GROUPS = [("W-GNN", W), ("G+LM", [("lmggnn", "Vul-LMGGNN"), ("regvd", "ReGVD")]),
          ("Trans", [("codebert", "CodeBERT$^*$"), ("codet5plus", "CodeT5+$^*$")]),
          ("N-N", [("tfidf_lr", "TF-IDF$+$LR"), ("cpg_lr", "CPG$+$LR")])]
SIG = ("\\textbf{bold} = statistically significant (neural: paired $t$-test over five seeds, BH FDR $q<0.05$ "
       "within the table; non-neural: 95\\% interval over 30 training subsamples excludes 0)")
VALID = "$\\Delta$ values are F1 changes relative to clean performance on the same functions, restricted to valid transformations"
cap = f"PrimeVul robustness results. Clean F1: mean\\,$\\pm$\\,std. {VALID}; {SIG}. $*$~sequence input."
L = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{cap}}}", "\\label{tab:primevul}",
     "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llrrrrrrrrr}", "\\toprule", HDR, "\\midrule"]
for gi, (fam, rows) in enumerate(GROUPS):
    if gi:
        L.append("\\midrule")
    for ri, (m, name) in enumerate(rows):
        s = st[m]; c0 = s["identifier"]
        cells = [num(s[c]["delta_f1"], bold=bool(s[c]["significant"])) for c in CONDS]
        mx = max(abs(s[c]["delta_f1"]) for c in CONDS)
        first = f"\\multirow{{{len(rows)}}}{{*}}{{{fam}}}\n" if ri == 0 else ""
        L.append(f"{first} & {name} & {c0['clean_f1_full_test']:.2f}\\,$\\pm$\\,{c0['clean_f1_full_test_sd']:.2f} & "
                 + " & ".join(cells) + f" & {mx:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}%", "}", "\\vspace{-2mm}", "\\end{table*}"]
(OUT / "table_primevul.tex").write_text("\n".join(L) + "\n")
print("wrote", OUT / "table_primevul.tex")
