"""DeepWukong case-study tables: detection deltas and per-instance pipeline instrumentation.

  python -m pipeline.deepwukong.report

Detection (per seed, threshold tuned on the validation split as for every detector):
  paired   F1 on test XFGs that exist in both clean and transformed form, whose key line lies in a validly
           transformed function. Ground truth is the clean label.
  e2e      end to end over all clean test XFGs; an XFG whose transformed counterpart was not produced
           (seed lost) counts as not flagged.
  Delta = transformed - clean on the same XFGs; paired t-test over seeds, BH across the table.
Instrumentation (seed-independent, from work/<cond>/index.jsonl): seed retained (all / vulnerable), slice
identical to clean, slice Jaccard (lines mapped back to the original), slices containing inserted lines,
recomputed-label flips (0->1 pollution, 1->0 hiding), node and edge change, new seeds per file.
Writes results/stats/deepwukong.json and results/tables/table_deepwukong{,_pipeline}.{tex,csv}.
"""
import glob
import gzip
import json

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from pipeline.deepwukong.common import CONDITIONS, ROOT, work

NAMES = {"identifier": "Ren", "deadcode": "Dead", "controlflow": "CF", "ren_dead": "R+D", "ren_cf": "R+CF",
         "dead_cf": "D+CF", "compound": "Cmp"}
NAMES.update({f"v1_{k}": f"{v} (v1 transforms)" for k, v in list(NAMES.items())})


def _m(y, p, t):
    y, p = np.asarray(y), np.asarray(p)
    yh = p >= t
    return {"f1": 100 * f1_score(y, yh, zero_division=0), "precision": 100 * precision_score(y, yh, zero_division=0),
            "recall": 100 * recall_score(y, yh, zero_division=0),
            "roc_auc": 100 * roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"), "n": len(y), "n_vul": int(y.sum())}


def _bh(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 1.0
    for k, i in enumerate(order[::-1]):
        rank = len(ps) - k
        run = min(run, ps[i] * len(ps) / rank)
        adj[i] = run
    return adj


def load_index(c):
    return {r["id"]: r for r in map(json.loads, open(work(c) / "index.jsonl"))}


def main():
    runs = [json.load(gzip.open(f)) for f in sorted(glob.glob(str(ROOT / "results/preds/deepwukong/deepwukong/run_*.json.gz")))]
    conds = [c for c in CONDITIONS[1:] + [f"v1_{c}" for c in CONDITIONS[1:]] if all(c in r["test"] for r in runs) and (work(c) / "index.jsonl").exists()]
    clean_idx = load_index("clean")
    res = {"n_runs": len(runs), "clean_shipped": [], "conditions": {}}
    for r in runs:
        cs = r["test"]["clean_shipped"]
        res["clean_shipped"].append(_m(cs["y"], cs["p"], r["threshold"]))
    for c in conds:
        idx = load_index(c)
        scope = {i for i, x in idx.items() if x["found"] and x["valid_scope"] and clean_idx.get(i, {}).get("found")}
        per = {"paired": [], "e2e": []}
        for r in runs:
            t, cl, tr = r["threshold"], r["test"]["clean"], r["test"][c]
            pc, pt = dict(zip(cl["ids"], cl["p"])), dict(zip(tr["ids"], tr["p"]))
            yc = dict(zip(cl["ids"], cl["y"]))
            ids = sorted(scope & pc.keys() & pt.keys())
            a, b = _m([yc[i] for i in ids], [pc[i] for i in ids], t), _m([yc[i] for i in ids], [pt[i] for i in ids], t)
            per["paired"].append({"clean": a, "trans": b})
            allids = sorted(pc)
            a2 = _m([yc[i] for i in allids], [pc[i] for i in allids], t)
            b2 = _m([yc[i] for i in allids], [pt.get(i, 0.0) for i in allids], t)
            per["e2e"].append({"clean": a2, "trans": b2})
        out = {}
        for scope_name, rows in per.items():
            d = {}
            for k in ["f1", "precision", "recall", "roc_auc"]:
                dc = np.array([x["trans"][k] - x["clean"][k] for x in rows])
                d[k] = {"clean_mean": float(np.mean([x["clean"][k] for x in rows])),
                        "trans_mean": float(np.mean([x["trans"][k] for x in rows])),
                        "delta_mean": float(dc.mean()), "delta_sd": float(dc.std(ddof=1)) if len(dc) > 1 else 0.0,
                        "p": float(stats.ttest_1samp(dc, 0).pvalue) if len(dc) > 1 and dc.std() > 0 else 1.0}
            d["n"], d["n_vul"] = rows[0]["clean"]["n"], rows[0]["clean"]["n_vul"]
            out[scope_name] = d
        # instrumentation
        vals = [x for i, x in idx.items() if i in clean_idx]
        found = [x for x in vals if x["found"]]
        vul = [x for x in vals if x["label"] == 1]
        jac, ident, ins, dn, de = [], 0, 0, [], []
        for x in found:
            c0 = clean_idx[x["id"]]
            if not c0.get("found"):
                continue
            s0, s1 = set(c0["slice_orig"]), set(x["slice_orig"])
            jac.append(len(s0 & s1) / max(1, len(s0 | s1)))
            ident += s0 == s1 and x["inserted_lines"] == 0
            ins += x["inserted_lines"] > 0
            dn.append(x["nodes"] - c0["nodes"])
            de.append(x["edges"] - c0["edges"])
        fs0 = json.loads((work("clean") / "file_stats.json").read_text())
        fs1 = json.loads((work(c) / "file_stats.json").read_text())
        out["pipeline"] = {
            "n_test_xfgs": len(vals), "valid_scope_pct": 100 * np.mean([x["valid_scope"] for x in vals]),
            "key_mapped_pct": 100 * np.mean([x["key_mapped"] for x in vals]),
            "seed_retained_pct": 100 * len(found) / len(vals),
            "vul_seed_retained_pct": 100 * np.mean([x["found"] for x in vul]),
            "slice_identical_pct": 100 * ident / max(1, len(found)), "slice_jaccard_mean": float(np.mean(jac)),
            "slices_with_inserted_lines_pct": 100 * ins / max(1, len(found)),
            "label_pollution_0to1": sum(x["label"] == 0 and x["label_recomputed"] == 1 for x in found),
            "label_hiding_1to0": sum(x["label"] == 1 and x["label_recomputed"] == 0 for x in found),
            "nodes_delta_mean": float(np.mean(dn)), "edges_delta_mean": float(np.mean(de)),
            "seeds_per_file_clean": float(np.mean([sum(v.values()) for v in fs0.values()])),
            "seeds_per_file_trans": float(np.mean([sum(fs1[k].values()) for k in fs0 if k in fs1])),
        }
        res["conditions"][c] = out
    keys = [(c, s) for c in conds for s in ("paired", "e2e")]
    adj = _bh([res["conditions"][c][s]["f1"]["p"] for c, s in keys]) if keys else []
    for (c, s), q in zip(keys, adj):
        res["conditions"][c][s]["f1"]["p_bh"] = float(q)
        res["conditions"][c][s]["f1"]["significant"] = bool(q < 0.05)
    (ROOT / "results/stats").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/stats/deepwukong.json").write_text(json.dumps(res, indent=1))
    write_tables(res, conds)


def write_tables(res, conds):
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    cs = np.array([x["f1"] for x in res["clean_shipped"]])
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Condition & $n$ (vul) & Clean F1 & $\Delta$F1 & $\Delta$Prec & $\Delta$Rec & $\Delta$F1 end-to-end \\", r"\midrule",
             rf"Clean (authors' test XFGs) & & {cs.mean():.2f}\,$\pm$\,{cs.std(ddof=1) if len(cs) > 1 else 0:.2f} & & & & \\"]
    csv = ["condition,n,n_vul,clean_f1,delta_f1,delta_f1_sd,p_bh,delta_precision,delta_recall,delta_roc_auc,e2e_delta_f1,e2e_p_bh"]
    fmt = lambda d: (r"$\mathbf{%+.2f}$" if d.get("significant") else r"$%+.2f$") % d["delta_mean"]  # noqa: E731
    for c in conds:
        p, e = res["conditions"][c]["paired"], res["conditions"][c]["e2e"]
        lines.append(f"{NAMES[c]} & {p['n']} ({p['n_vul']}) & {p['f1']['clean_mean']:.2f} & {fmt(p['f1'])} & "
                     f"${p['precision']['delta_mean']:+.2f}$ & ${p['recall']['delta_mean']:+.2f}$ & {fmt(e['f1'])} \\\\")
        csv.append(f"{c},{p['n']},{p['n_vul']},{p['f1']['clean_mean']:.4f},{p['f1']['delta_mean']:.4f},{p['f1']['delta_sd']:.4f},"
                   f"{p['f1']['p_bh']:.4g},{p['precision']['delta_mean']:.4f},{p['recall']['delta_mean']:.4f},"
                   f"{p['roc_auc']['delta_mean']:.4f},{e['f1']['delta_mean']:.4f},{e['f1']['p_bh']:.4g}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_deepwukong.tex").write_text("\n".join(lines) + "\n")
    (out / "table_deepwukong.csv").write_text("\n".join(csv) + "\n")
    cols = ["seed_retained_pct", "vul_seed_retained_pct", "slice_identical_pct", "slice_jaccard_mean",
            "slices_with_inserted_lines_pct", "label_pollution_0to1", "label_hiding_1to0", "nodes_delta_mean",
            "edges_delta_mean", "seeds_per_file_clean", "seeds_per_file_trans", "valid_scope_pct"]
    head = ["Condition", "Seed kept \\%", "Vul seed kept \\%", "Slice same \\%", "Jaccard", "Inserted \\%",
            "0$\\to$1", "1$\\to$0", "$\\Delta$nodes", "$\\Delta$edges", "Seeds/file", "Valid \\%"]
    t = [r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    pc = ["condition," + ",".join(cols)]
    for c in conds:
        q = res["conditions"][c]["pipeline"]
        cells = [f"{q[k]:.1f}" if isinstance(q[k], float) else str(q[k]) for k in cols]
        cells[10] = f"{q['seeds_per_file_clean']:.1f}$\\to${q['seeds_per_file_trans']:.1f}"
        del cells[9]
        t.append(f"{NAMES[c]} & " + " & ".join(cells) + r" \\")
        pc.append(c + "," + ",".join(f"{q[k]}" for k in cols))
    t += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_deepwukong_pipeline.tex").write_text("\n".join(t).replace(" & Seeds/file & Valid", " & Seeds/file & Valid") + "\n")
    (out / "table_deepwukong_pipeline.csv").write_text("\n".join(pc) + "\n")
    print("tables ->", out)


if __name__ == "__main__":
    main()
