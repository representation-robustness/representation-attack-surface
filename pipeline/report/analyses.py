"""Paper tables beyond the robustness tables, all computed from result files (no hand-copied numbers).

  python -m pipeline.report.analyses          # run after pipeline.report.refresh

  table_ablation_<ds>      identifier-exposure ablation: full model vs node-types-only (-S) and
                           VulGNN-S + CodeBERT; clean / Ren F1 on the identically renamed subset, Delta Ren
                           with BH-adjusted p, and the difference in Delta Ren (Welch t over seeds)
  table_minbudget_<ds>     oracle minimum-budget audit (ASR@k, mean budget, edit cost of the chosen
                           sequence) and the realisable fixed-policy ASR, with denominators
  table_transfer_<ds>      TASR matrix (mean over paired runs) with source-evaded denominators in the csv
  table_mitigation_devign  clean / volume-control / augmented / held-out-family training
  table_flips_<ds>         instance-level prediction flips per condition (non-vul -> flagged, vul -> missed)
  table_cf_sensitivity_<ds> control-flow conditions with fallback-only rewrites excluded
  table_edit_cost          tokens / identifiers / sites and graph edit size per condition; why invalid ones failed
  table_pattern_density    surface-pattern occurrences per 1,000 training functions, by class
  correlation.json         clean F1 vs Max|Delta| across the main detectors (Pearson and Spearman)
  fig_heatmap_<ds>.pdf     Delta F1 heatmap, main detectors x conditions
Everything is written to results/tables/ (tex + csv) except figures (results/figures/).
"""
import gzip
import json
import re

import numpy as np
from scipy import stats

from pipeline.models.common import CONDITIONS, ROOT
from pipeline.report.tables import COND_SHORT, DATASETS, NAMES

OUT = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
MAIN = ["ecg_rgcn", "angle", "vulgnn", "reveal", "lmggnn", "regvd", "codebert", "codet5plus", "tfidf_lr", "cpg_lr"]
NAMES = dict(NAMES, codebert_aug="CodeBERT-Aug", regvd_aug="ReGVD-Aug", reveal_aug="REVEAL-Aug")


def metrics(ds, model):
    p = ROOT / "results" / "metrics" / ds / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stat_rows(ds, scope="valid"):
    p = ROOT / "results" / "stats" / f"{ds}_{scope}.json"
    return {(r["model"], r["condition"]): r for r in json.loads(p.read_text())} if p.exists() else {}


def write(name, tex_lines, csv_lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(tex_lines) + "\n")
    (OUT / f"{name}.csv").write_text("\n".join(csv_lines) + "\n")


def sig(txt, row):
    return f"$\\mathbf{{{txt}}}$" if row and row.get("significant") else f"${txt}$"


# ------------------------------------------------------------------ identifier-exposure ablation

ABLATION = [("reveal", "reveal_struct"), ("ecg_rgcn", "ecg_rgcn_struct"), ("angle", "angle_struct"),
            ("vulgnn", "vulgnn_struct"), ("vulgnn_struct", "vulgnn_struct_cb")]


def ren_per_run(m):
    """Per-run clean and renamed F1 on the validly renamed subset."""
    cl = np.array([r["valid"]["identifier"]["clean"]["f1"] for r in m["per_run"]])
    tr = np.array([r["valid"]["identifier"]["transformed"]["f1"] for r in m["per_run"]])
    return cl, tr


def ablation_table(ds):
    st = stat_rows(ds)
    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Architecture & Variant & Clean F1 & Ren F1 & $\Delta$Ren & $q_{BH}$ & Diff. vs.\ base ($p$) \\", r"\midrule"]
    csv = ["dataset,base,variant,n_runs,clean_f1,clean_sd,ren_f1,ren_sd,delta_ren,delta_ren_sd,p_bh,diff_vs_base,diff_p_welch"]
    for base, var in ABLATION:
        mb, mv = metrics(ds, base), metrics(ds, var)
        if not mb or not mv:
            continue
        db = None
        for k, (model, m) in enumerate(((base, mb), (var, mv))):
            cl, tr = ren_per_run(m)
            d = tr - cl
            row = st.get((model, "identifier"), {})
            q = row.get("p_bh")
            if k == 0:
                db, diff, dp = d, "", ""
                diff_tex = ""
            else:
                t, dp = stats.ttest_ind(d, db, equal_var=False) if d.std() + db.std() > 0 else (0.0, 1.0)
                diff = d.mean() - db.mean()
                diff_tex = f"${diff:+.2f}$ ({dp:.2g})"
            label = "node types only" if model.endswith("_struct") else ("+ CodeBERT [CLS]" if model.endswith("_cb") else "full")
            arch = NAMES[base.replace("_struct", "")] if k == 0 else ""
            dtxt = sig(f"{d.mean():+.2f}", row)
            qtxt = "--" if q is None else f"{q:.3g}"
            tex.append(f"{arch} & {label} & {cl.mean():.2f}\\,$\\pm$\\,{cl.std(ddof=1):.2f} & "
                       f"{tr.mean():.2f}\\,$\\pm$\\,{tr.std(ddof=1):.2f} & {dtxt} & {qtxt} & {diff_tex} \\\\")
            csv.append(f"{ds},{base},{model},{len(d)},{cl.mean():.4f},{cl.std(ddof=1):.4f},{tr.mean():.4f},{tr.std(ddof=1):.4f},"
                       f"{d.mean():.4f},{d.std(ddof=1):.4f},{'' if q is None else f'{q:.4g}'},"
                       f"{'' if diff == '' else f'{diff:.4f}'},{'' if dp == '' else f'{dp:.4g}'}")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    write(f"table_ablation_{ds}", tex, csv)


# ------------------------------------------------------------------ minimum budget / fixed policy

ATTACK_ROWS = ["ecg_rgcn", "angle", "vulgnn", "reveal", "lmggnn", "regvd", "codebert", "codet5plus", "tfidf_lr",
               "cpg_lr", "ecg_rgcn_struct", "angle_struct", "vulgnn_struct", "reveal_struct"]


def minbudget_table(ds):
    tex = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
           r"System & $|X_f|$ & ASR@1 & ASR@2 & ASR@3 & Mean $k$ & Ids & Tokens & Fixed Ren & Fixed Dead & Fixed CF \\",
           r"\midrule"]
    csv = ["dataset,model,n_runs,denominator_mean,asr1,asr1_sd,asr2,asr2_sd,asr3,asr3_sd,mean_budget,"
           "identifiers_renamed,occurrences_renamed,sites,tokens_added," + ",".join(f"fixed_{c},fixed_{c}_denominator" for c in CONDITIONS)]
    for m in ATTACK_ROWS:
        p = ROOT / "results" / "attacks" / ds / f"minbudget_{m}.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        pr = a["per_run"]
        den = np.mean([r["denominator"] for r in pr])
        asr = {k: (np.mean([r["asr_at_k"][str(k)] if str(k) in r["asr_at_k"] else r["asr_at_k"][k] for r in pr]),
                   np.std([r["asr_at_k"][str(k)] if str(k) in r["asr_at_k"] else r["asr_at_k"][k] for r in pr], ddof=1) if len(pr) > 1 else 0.0)
               for k in (1, 2, 3)}
        mb = np.mean([r["mean_min_budget"] for r in pr if r["mean_min_budget"] is not None]) if any(r["mean_min_budget"] for r in pr) else float("nan")
        cost = {k: np.mean([r["mean_cost_of_min_sequence"].get(k, 0) for r in pr if r["mean_cost_of_min_sequence"]] or [0])
                for k in ("identifiers_renamed", "occurrences_renamed", "sites", "tokens_added")}
        fixed = {c: (np.mean([r["fixed_policy"][c]["asr"] for r in pr]), np.mean([r["fixed_policy"][c]["denominator"] for r in pr]))
                 for c in CONDITIONS}
        tex.append(f"{NAMES.get(m, m)} & {den:.0f} & " + " & ".join(f"{asr[k][0]:.1f}\\,$\\pm$\\,{asr[k][1]:.1f}" for k in (1, 2, 3)) +
                   f" & {mb:.2f} & {cost['identifiers_renamed']:.1f} & {cost['tokens_added']:.1f} & " +
                   " & ".join(f"{fixed[c][0]:.1f}" for c in CONDITIONS[:3]) + r" \\")
        csv.append(f"{ds},{m},{len(pr)},{den:.1f}," + ",".join(f"{asr[k][0]:.4f},{asr[k][1]:.4f}" for k in (1, 2, 3)) +
                   f",{mb:.4f}," + ",".join(f"{cost[k]:.4f}" for k in cost) + "," +
                   ",".join(f"{fixed[c][0]:.4f},{fixed[c][1]:.1f}" for c in CONDITIONS))
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(f"table_minbudget_{ds}", tex, csv)


TRANSFER = ["angle", "vulgnn", "ecg_rgcn", "reveal", "regvd", "lmggnn", "codebert", "codet5plus", "tfidf_lr", "cpg_lr"]


def transfer_table(ds):
    p = ROOT / "results" / "attacks" / ds / "transfer.json"
    if not p.exists():
        return
    tr = {(x["source"], x["target"]): x for x in json.loads(p.read_text())}
    ms = [m for m in TRANSFER if any(k[0] == m for k in tr)]
    tex = [r"\begin{tabular}{l" + "r" * len(ms) + "}", r"\toprule",
           r"Source $\rightarrow$ Target & " + " & ".join(NAMES.get(m, m) for m in ms) + r" \\", r"\midrule"]
    csv = ["dataset,source,target,n_pairs,tasr_mean,tasr_sd,source_evaded_mean,joint_correct_mean,transferred_mean"]
    for s in ms:
        cells = []
        for t in ms:
            x = tr.get((s, t))
            if s == t or not x or x["tasr_mean"] is None:
                cells.append("--")
                continue
            cells.append(f"{x['tasr_mean']:.1f}")
            pr = x["per_run"]
            csv.append(f"{ds},{s},{t},{len(pr)},{x['tasr_mean']:.4f},{x['tasr_sd']:.4f},"
                       f"{np.mean([r['source_evaded'] for r in pr]):.1f},{np.mean([r['joint_correct'] for r in pr]):.1f},"
                       f"{np.mean([r['transferred'] for r in pr]):.1f}")
        tex.append(f"{NAMES.get(s, s)} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(f"table_transfer_{ds}", tex, csv)


# ------------------------------------------------------------------ robustness training (Devign)

def mitigation_table(ds="devign"):
    st = stat_rows(ds)
    held = {"aug_dead+cf": "identifier", "aug_ren+cf": "deadcode", "aug_ren+dead": "controlflow"}
    tex = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Model & Training data & Clean F1 & $\Delta$Ren & $\Delta$Dead & $\Delta$CF & Max$|\Delta|$ & Held-out $\Delta$ \\", r"\midrule"]
    csv = ["dataset,model,variant,n_runs,clean_f1,clean_sd," + ",".join(f"delta_{c},p_bh_{c}" for c in CONDITIONS) + ",max_abs_delta,held_out_family,held_out_delta"]
    for base in ("codebert", "regvd", "reveal"):
        variants = [(base, "clean"), (f"{base}_volctl", "clean $\\times$4 (volume control)"), (f"{base}_aug", "clean + Ren + Dead + CF")]
        variants += [(f"{base}_{k}", f"clean + {k[4:].replace('ren', 'Ren').replace('dead', 'Dead').replace('cf', 'CF')}") for k in held]
        first = True
        for m, label in variants:
            if (m, "identifier") not in st:
                continue
            r0 = st[(m, "identifier")]
            ds_ = {c: st.get((m, c)) for c in CONDITIONS}
            mx = max(abs(v["delta_f1"]) for v in ds_.values() if v)
            hk = next((k for k in held if m.endswith(k)), None)
            ho = ds_[held[hk]] if hk else None
            tex.append(f"{NAMES[base] if first else ''} & {label} & {r0['clean_f1_full_test']:.2f}\\,$\\pm$\\,{r0['clean_f1_full_test_sd']:.2f} & " +
                       " & ".join(sig(f"{ds_[c]['delta_f1']:+.2f}", ds_[c]) for c in CONDITIONS[:3]) +
                       f" & {mx:.2f} & " + (sig(f"{ho['delta_f1']:+.2f}", ho) + f" ({COND_SHORT[CONDITIONS.index(held[hk])]})" if ho else "--") + r" \\")
            csv.append(f"{ds},{base},{m},{r0['n_runs']},{r0['clean_f1_full_test']:.4f},{r0['clean_f1_full_test_sd']:.4f}," +
                       ",".join(f"{ds_[c]['delta_f1']:.4f},{ds_[c].get('p_bh', '')}" for c in CONDITIONS) +
                       f",{mx:.4f},{held[hk] if hk else ''},{ho['delta_f1'] if ho else ''}")
            first = False
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    write(f"table_mitigation_{ds}", tex, csv)


# ------------------------------------------------------------------ instance-level flips

def flips_table(ds):
    """For each run: on validly transformed test functions, the share of clean-negative non-vulnerable
    functions that become flagged (FP introduced) and of clean-detected vulnerable functions that become
    missed (FN introduced), plus the reverse flips. Mean over runs."""
    from pipeline.eval.metrics import valid_ids
    csv = ["dataset,model,condition,n_runs,nonvul_neg_to_pos_pct,nonvul_pos_to_neg_pct,vul_pos_to_neg_pct,vul_neg_to_pos_pct,"
           "n_nonvul,n_vul,delta_precision,delta_recall,delta_f1"]
    st = stat_rows(ds)
    tex = [r"\begin{tabular}{l" + "rr" * 3 + "}", r"\toprule",
           r"System & \multicolumn{2}{c}{Dead} & \multicolumn{2}{c}{CF} & \multicolumn{2}{c}{Ren} \\",
           r" & NV$\uparrow$ & V$\downarrow$ & NV$\uparrow$ & V$\downarrow$ & NV$\uparrow$ & V$\downarrow$ \\", r"\midrule"]
    for m in MAIN:
        runs = sorted((ROOT / "results" / "preds" / ds / m).glob("run_*.json.gz"))
        if not runs:
            continue
        acc = {c: [] for c in CONDITIONS}
        for f in runs:
            r = json.load(gzip.open(f))
            thr, cl = r["threshold"], r["test"]["clean"]
            pc = {i: p >= thr for i, p in zip(cl["ids"], cl["p"])}
            y = dict(zip(cl["ids"], cl["y"]))
            for c in CONDITIONS:
                if c not in r["test"]:
                    continue
                keep = valid_ids(ds, c)
                pt = {i: p >= thr for i, p in zip(r["test"][c]["ids"], r["test"][c]["p"]) if i in keep and i in pc}
                nv = [i for i in pt if y[i] == 0]
                v = [i for i in pt if y[i] == 1]
                pct = lambda ids, a, b: 100 * sum(pc[i] == a and pt[i] == b for i in ids) / max(1, sum(pc[i] == a for i in ids))  # noqa: E731
                acc[c].append((pct(nv, False, True), pct(nv, True, False), pct(v, True, False), pct(v, False, True), len(nv), len(v)))
        cells = []
        for c in CONDITIONS:
            if not acc[c]:
                continue
            a = np.array(acc[c])
            s = st.get((m, c), {})
            csv.append(f"{ds},{m},{c},{len(a)}," + ",".join(f"{x:.3f}" for x in a.mean(0)[:4]) +
                       f",{a[0, 4]:.0f},{a[0, 5]:.0f},{s.get('delta_precision', float('nan')):.4f},{s.get('delta_recall', float('nan')):.4f},"
                       f"{s.get('delta_f1', float('nan')):.4f}")
        for c in ("deadcode", "controlflow", "identifier"):
            a = np.array(acc[c]) if acc[c] else None
            cells += [f"{a[:, 0].mean():.1f}", f"{a[:, 2].mean():.1f}"] if a is not None else ["--", "--"]
        tex.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(f"table_flips_{ds}", tex, csv)


# ------------------------------------------------------------------ control-flow sensitivity

def cf_fallback_ids(ds, cond):
    """Valid ids of `cond` whose control-flow step applied only the fallback (an inserted always-true
    guard block, no branch split / loop predicate / call wrap / flattening)."""
    out = set()
    for line in open(ROOT / "data" / ds / "validity" / "test" / f"{cond}.jsonl"):
        r = json.loads(line)
        cf = [t for t in r.get("transforms", []) if t.get("transform") == "controlflow"]
        if r["valid"] and cf and cf[0].get("fallback"):
            out.add(r["id"])
    return out


def cf_sensitivity_table(ds):
    """Delta F1 of control-flow conditions with fallback-only functions excluded, next to the main value."""
    from pipeline.eval.metrics import scores, valid_ids
    conds = ["controlflow", "ren_cf", "dead_cf", "compound"]
    st = stat_rows(ds)
    rows, pv = [], []
    for m in MAIN:
        runs = sorted((ROOT / "results" / "preds" / ds / m).glob("run_*.json.gz"))
        if not runs:
            continue
        data = [json.load(gzip.open(f)) for f in runs]
        for c in conds:
            keep = valid_ids(ds, c) - cf_fallback_ids(ds, c)
            d = []
            for r in data:
                t, cl, tr = r["threshold"], r["test"]["clean"], r["test"][c]
                ids = keep & set(cl["ids"]) & set(tr["ids"])
                pc, pt, y = dict(zip(cl["ids"], cl["p"])), dict(zip(tr["ids"], tr["p"])), dict(zip(cl["ids"], cl["y"]))
                ids = sorted(ids)
                d.append(scores([y[i] for i in ids], [pt[i] for i in ids], t)["f1"] - scores([y[i] for i in ids], [pc[i] for i in ids], t)["f1"])
            d = np.array(d)
            p = stats.ttest_1samp(d, 0).pvalue if m not in ("tfidf_lr", "cpg_lr") and d.std() > 0 else None
            lo, hi = np.percentile(d, [2.5, 97.5]) if m in ("tfidf_lr", "cpg_lr") else (None, None)
            rows.append({"model": m, "condition": c, "n": len(ids), "delta": float(d.mean()), "p": p, "lo": lo, "hi": hi,
                         "main_delta": st.get((m, c), {}).get("delta_f1")})
    from pipeline.eval.stats import bh
    fam = [r for r in rows if r["p"] is not None]
    for r, q in zip(fam, bh([r["p"] for r in fam])):
        r["q"] = q
    csv = ["dataset,model,condition,n_excluding_fallback,delta_f1_excluding_fallback,p_bh,delta_f1_main"]
    tex = [r"\begin{tabular}{l" + "rr" * len(conds) + "}", r"\toprule",
           "System & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{COND_SHORT[CONDITIONS.index(c)]}}}" for c in conds) + r" \\",
           " & " + " & ".join(["all & no fallback"] * len(conds)) + r" \\", r"\midrule"]
    by = {(r["model"], r["condition"]): r for r in rows}
    for m in dict.fromkeys(r["model"] for r in rows):
        cells = []
        for c in conds:
            r = by[(m, c)]
            sigr = {"significant": (r.get("q", 1) < 0.05) if r["p"] is not None else (r["lo"] > 0 or r["hi"] < 0)}
            cells += [sig(f"{r['main_delta']:+.2f}", st.get((m, c))), sig(f"{r['delta']:+.2f}", sigr)]
            csv.append(f"{ds},{m},{c},{r['n']},{r['delta']:.4f},{r.get('q', '')},{r['main_delta']}")
        tex.append(f"{NAMES.get(m, m)} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    write(f"table_cf_sensitivity_{ds}", tex, csv)


# ------------------------------------------------------------------ transformation cost and rejection reasons

def edit_cost_table():
    """Per dataset x condition, over validly transformed test functions: tokens added, identifiers renamed,
    edit sites (from the validity records) and graph edit size (node / edge / CFG-edge count change of the
    Joern graph, absolute and relative), plus why invalid transformations were rejected."""
    import collections
    from pipeline.eval.attacks import validity

    def sizes(ds, cond):
        out = {}
        with gzip.open(ROOT / "data" / ds / "graphs" / "test" / f"{cond}.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = (len(r["nodes"]), len(r["edges"]), sum(e[2] == "CFG" for e in r["edges"]))
        return out

    csv = ["dataset,condition,n_valid,tokens_added,identifiers_renamed,occurrences_renamed,sites,"
           "n_graph_pairs,d_nodes,d_nodes_pct,d_edges,d_edges_pct,d_cfg_edges"]
    rej = ["dataset,condition,n_test,unchanged,new_parse_errors,new_compile_errors,top_compile_error_kinds"]
    tex = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Dataset & Condition & Tokens & Ids & Sites & $\Delta$nodes\,\% & $\Delta$edges\,\% & $\Delta$CFG \\", r"\midrule"]
    for ds, nice in DATASETS.items():
        base = sizes(ds, "clean")
        for k, c in enumerate(CONDITIONS):
            v = validity(ds, c)
            ok = {i: cost for i, (valid, cost) in v.items() if valid}
            g = sizes(ds, c)
            pairs = [i for i in ok if i in g and i in base]
            dn = np.array([g[i][0] - base[i][0] for i in pairs]); de = np.array([g[i][1] - base[i][1] for i in pairs])
            dc = np.array([g[i][2] - base[i][2] for i in pairs])
            rn = np.array([100 * (g[i][0] - base[i][0]) / max(1, base[i][0]) for i in pairs])
            re_ = np.array([100 * (g[i][1] - base[i][1]) / max(1, base[i][1]) for i in pairs])
            mc = {key: np.mean([x[key] for x in ok.values()]) for key in ("tokens_added", "identifiers_renamed", "occurrences_renamed", "sites")}
            csv.append(f"{ds},{c},{len(ok)}," + ",".join(f"{mc[x]:.2f}" for x in mc) +
                       f",{len(pairs)},{dn.mean():.2f},{rn.mean():.2f},{de.mean():.2f},{re_.mean():.2f},{dc.mean():.2f}")
            tex.append(f"{nice if k == 0 else ''} & {COND_SHORT[k]} & {mc['tokens_added']:.1f} & {mc['identifiers_renamed']:.1f} & "
                       f"{mc['sites']:.1f} & {rn.mean():+.1f} & {re_.mean():+.1f} & {dc.mean():+.1f} \\\\")
            n = unch = pe = ce = 0
            kinds = collections.Counter()
            for line in open(ROOT / "data" / ds / "validity" / "test" / f"{c}.jsonl"):
                r = json.loads(line)
                n += 1
                if r["valid"]:
                    continue
                if not r.get("changed"):
                    unch += 1
                elif r.get("new_parse_errors"):
                    pe += 1
                else:
                    ce += 1
                    kinds.update(r.get("new_compile_error_kinds", []))
            rej.append(f"{ds},{c},{n},{unch},{pe},{ce}," + ";".join(f"{x}:{m}" for x, m in kinds.most_common(3)).replace(",", " "))
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}")
    write("table_edit_cost", tex, csv)
    (OUT / "table_validity_rejections.csv").write_text("\n".join(rej) + "\n")


# ------------------------------------------------------------------ surface pattern density

PATTERNS = {  # regexes over comment- and string-stripped source
    "if": r"\bif\s*\(", "while": r"\bwhile\s*\(", "for": r"\bfor\s*\(",
    "alloc": r"\b\w*(?:malloc|calloc|realloc|alloc)\s*\(", "memset": r"\bmemset\s*\(", "memcpy": r"\bmemcpy\s*\(",
    "error": r"\b\w*err(?:or)?\w*\b", "printf": r"\b\w*printf\s*\(",
}
STRIP = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", re.S)


def pattern_density():
    from pipeline.models.common import load_text
    rx = {k: re.compile(v) for k, v in PATTERNS.items()}
    res = {}
    for ds in DATASETS:
        rows = load_text(ds, "train")
        key = next(k for k in ("func", "code", "source") if k in rows[0])
        for cls in (1, 0):
            fs = [STRIP.sub(" ", r[key]) for r in rows if r["target"] == cls]
            res[(ds, cls)] = {k: 1000 * sum(len(r.findall(f)) for f in fs) / len(fs) for k, r in rx.items()} | {"n": len(fs)}
    tex = [r"\begin{tabular}{l" + "rr" * len(DATASETS) + "}", r"\toprule",
           "Pattern & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in DATASETS.values()) + r" \\",
           " & " + " & ".join(["Vul & Non-vul"] * len(DATASETS)) + r" \\", r"\midrule"]
    csv = ["pattern," + ",".join(f"{ds}_vul,{ds}_nonvul" for ds in DATASETS)]
    for k in PATTERNS:
        vals = [res[(ds, c)][k] for ds in DATASETS for c in (1, 0)]
        tex.append(f"\\texttt{{{k}}} & " + " & ".join(f"{v:,.0f}" for v in vals) + r" \\")
        csv.append(k + "," + ",".join(f"{v:.2f}" for v in vals))
    csv.append("n_functions," + ",".join(str(res[(ds, c)]["n"]) for ds in DATASETS for c in (1, 0)))
    tex += [r"\bottomrule", r"\end{tabular}"]
    write("table_pattern_density", tex, csv)


# ------------------------------------------------------------------ correlation and heatmap

def correlation():
    out = {}
    for ds in DATASETS:
        st = stat_rows(ds)
        pts = []
        for m in MAIN:
            if (m, "identifier") not in st:
                continue
            mx = max(abs(st[(m, c)]["delta_f1"]) for c in CONDITIONS if (m, c) in st)
            pts.append((m, st[(m, "identifier")]["clean_f1_full_test"], mx))
        if len(pts) < 3:
            continue
        x, y = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
        pr, sr = stats.pearsonr(x, y), stats.spearmanr(x, y)
        out[ds] = {"n_detectors": len(pts), "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
                   "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
                   "max_abs_delta_range": [float(y.min()), float(y.max())],
                   "points": [{"model": m, "clean_f1": float(a), "max_abs_delta": float(b)} for m, a, b in pts]}
    (OUT / "correlation.json").write_text(json.dumps(out, indent=1))


def heatmaps():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for ds, nice in DATASETS.items():
        st = stat_rows(ds)
        ms = [m for m in MAIN if (m, "identifier") in st]
        if not ms:
            continue
        a = np.array([[st[(m, c)]["delta_f1"] if (m, c) in st else np.nan for c in CONDITIONS] for m in ms])
        lim = np.nanmax(np.abs(a))
        fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(ms) + 0.9))
        im = ax.imshow(a, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(CONDITIONS)), COND_SHORT)
        ax.set_yticks(range(len(ms)), [NAMES.get(m, m) for m in ms])
        for i, m in enumerate(ms):
            for j, c in enumerate(CONDITIONS):
                if not np.isnan(a[i, j]):
                    bold = st[(m, c)].get("significant")
                    ax.text(j, i, f"{a[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                            fontweight="bold" if bold else "normal", color="white" if abs(a[i, j]) > 0.6 * lim else "black")
        fig.colorbar(im, ax=ax, label="$\\Delta$F1 (pp)")
        ax.set_title(f"{nice}: $\\Delta$F1 on validly transformed functions", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_heatmap_{ds}.pdf")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if not (ROOT / "results" / "stats" / f"{ds}_valid.json").exists():
            continue
        ablation_table(ds)
        minbudget_table(ds)
        transfer_table(ds)
        flips_table(ds)
        cf_sensitivity_table(ds)
    mitigation_table("devign")
    pattern_density()
    edit_cost_table()
    correlation()
    heatmaps()
    print("analyses ->", OUT, FIG)


if __name__ == "__main__":
    main()
