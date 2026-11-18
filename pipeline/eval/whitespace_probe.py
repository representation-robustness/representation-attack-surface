"""Formatting-shortcut probe for the sequence detectors (CodeBERT, CodeT5+).

Motivation: on Big-Vul the transformer detectors reach much higher clean F1 than the
whole-function GNNs. This probe tests how much of that F1 rests on source formatting
(indentation / whitespace) rather than program content, by perturbing only the
whitespace of the test functions and re-scoring the fixed checkpoint. It addresses the graph-vs-transformer clean-F1 gap on Big-Vul.

Four whitespace-only perturbations, each semantics- and token-preserving in meaning:
  strip_all_indent   remove leading indentation on every line
  strip_trailing_ws  remove trailing whitespace before each newline
  double_newlines    double every newline
  reindent_4sp_clean tabs->4 spaces and strip trailing whitespace

Run, per seed (GPU, inference only; no training):
  python -m pipeline.eval.whitespace_probe run --seed 42
      -> appends one JSON record per (dataset, model) to
         $RERUN_ROOT/results/whitespace_probe/raw.jsonl

Aggregate over the recorded seeds (CPU):
  python -m pipeline.eval.whitespace_probe aggregate
      -> $RERUN_ROOT/results/tables/table_whitespace_probe.{csv,tex}
         (mean +/- std of Delta F1 across seeds)

The committed record is this script; the raw per-seed JSONL and the table are
regenerated under $RERUN_ROOT like every other result.
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from pipeline.models.common import HF_REVISIONS, ROOT, best_threshold, load_text

DATASETS = ["devign", "diversevul", "bigvul"]
MODELS = ["codebert", "codet5plus"]
PERTURB = {
    "strip_all_indent":   lambda s: re.sub(r"(?m)^[ \t]+", "", s),
    "strip_trailing_ws":  lambda s: re.sub(r"[ \t]+(?=\n)", "", s),
    "double_newlines":    lambda s: s.replace("\n", "\n\n"),
    "reindent_4sp_clean": lambda s: "\n".join(l.rstrip() for l in s.replace("\t", "    ").split("\n")),
}
OUT_DIR = ROOT / "results" / "whitespace_probe"
RAW = OUT_DIR / "raw.jsonl"


def run(seed: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from pipeline.models.transformer import CFG, Classifier, encode, predict

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW, "a") as out:
        for ds in DATASETS:
            test = load_text(ds, "test")
            y = np.array([r["target"] for r in test])
            valid = load_text(ds, "valid")
            yv = [r["target"] for r in valid]
            for mn in MODELS:
                c = CFG[mn]
                tok = AutoTokenizer.from_pretrained(c["pretrained"], revision=HF_REVISIONS[c["pretrained"]])
                m = Classifier(c["pretrained"], c["pooling"]).to(dev)
                m.load_state_dict(torch.load(ROOT / "runs/ckpt" / ds / mn / f"seed{seed}.pt", map_location=dev))
                score = lambda recs: predict(m, encode(tok, recs, c["max_len"]), 64, dev)
                thr = best_threshold(yv, score(valid))
                pc = score(test)
                rec = {"seed": seed, "dataset": ds, "model": mn,
                       "clean_f1": round(100 * f1_score(y, pc >= thr), 4),
                       "clean_auc": round(100 * roc_auc_score(y, pc), 4)}
                for k, fn in PERTURB.items():
                    p = score([dict(r, func=fn(r["func"])) for r in test])
                    rec[f"d_f1_{k}"] = round(100 * f1_score(y, p >= thr) - rec["clean_f1"], 4)
                    rec[f"d_auc_{k}"] = round(100 * roc_auc_score(y, p) - rec["clean_auc"], 4)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(json.dumps(rec), flush=True)
                del m
                torch.cuda.empty_cache()


def aggregate() -> None:
    recs = [json.loads(l) for l in open(RAW)]
    by = defaultdict(list)
    for r in recs:
        by[(r["dataset"], r["model"])].append(r)
    tables = (ROOT / "results" / "tables")
    tables.mkdir(parents=True, exist_ok=True)
    cols = list(PERTURB)
    header = "dataset,model,n_seeds,clean_f1," + ",".join(f"d_f1_{k},d_f1_{k}_sd" for k in cols)
    csv = [header]
    tex = [r"\begin{tabular}{llr" + "r" * len(cols) + "}", r"\toprule",
           "Dataset & Model & Clean F1 & " + " & ".join(k.replace("_", "\\_") for k in cols) + r" \\",
           r"\midrule"]
    for (ds, mn), rs in sorted(by.items()):
        n = len(rs)
        cf = np.mean([r["clean_f1"] for r in rs])
        cells, texcells = [], []
        for k in cols:
            v = np.array([r[f"d_f1_{k}"] for r in rs])
            cells += [f"{v.mean():.4f}", f"{v.std(ddof=1) if n > 1 else 0.0:.4f}"]
            texcells.append(f"${v.mean():+.1f}$" + (f"\\,$\\pm$\\,{v.std(ddof=1):.1f}" if n > 1 else ""))
        csv.append(f"{ds},{mn},{n},{cf:.4f}," + ",".join(cells))
        tex.append(f"{ds} & {mn} & {cf:.1f} & " + " & ".join(texcells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_whitespace_probe.csv").write_text("\n".join(csv) + "\n")
    (tables / "table_whitespace_probe.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {tables/'table_whitespace_probe.csv'} and .tex from {len(recs)} records "
          f"({len(by)} dataset/model cells)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seed)
    else:
        aggregate()


if __name__ == "__main__":
    main()
