"""DeepWukong end-to-end detection per function (fixed original function population).

Population: original SARD test functions (tree-sitter spans in work/clean/functions.jsonl) that contain the
key line of at least one shipped test XFG. XFGs whose key line lies outside any function are excluded.
Label: a function is vulnerable if any of its test XFGs is labelled vulnerable.
Aggregation rule (gate view): a function is flagged if any of its test XFGs scores at or above the run's
validation threshold. Under a transformation, an XFG that is no longer produced counts as not flagged;
slices newly created by the transformation are not scored (they have no clean counterpart).
Reported per condition: function-level F1 / precision / recall (mean over seeds), flagged vulnerable
functions, and Delta F1 vs. clean (mean, sd over seeds).

  python -m pipeline.deepwukong.function_level
Writes results/stats/deepwukong_function_level.json.
"""
import collections
import glob
import gzip
import json

import numpy as np

from pipeline.deepwukong.common import BASE, CONDITIONS, ROOT, work


def main():
    ti = json.load(open(BASE / "test_index.json"))
    spans = collections.defaultdict(list)
    for r in map(json.loads, open(work("clean") / "functions.jsonl")):
        spans[r["file"]].append((r["start"], r["end"]))
    fn, outside = {}, 0
    for x in ti:
        k = [(s, e) for s, e in spans[x["file"]] if s <= x["key_line"] <= e]
        if k:
            fn[x["id"]] = (x["file"],) + min(k, key=lambda z: z[1] - z[0])
        else:
            outside += 1
    lab = collections.defaultdict(int)
    for x in ti:
        if x["id"] in fn:
            lab[fn[x["id"]]] |= x["label"]
    funcs = sorted(lab)
    y = np.array([lab[f] for f in funcs])
    per = {c: [] for c in CONDITIONS}
    for path in sorted(glob.glob(str(ROOT / "results/preds/deepwukong/deepwukong/run_*.json.gz"))):
        d = json.load(gzip.open(path))
        t = d["threshold"]
        for c in CONDITIONS:
            flag = collections.defaultdict(int)
            for i, p in zip(d["test"][c]["ids"], d["test"][c]["p"]):
                if i in fn and p >= t:
                    flag[fn[i]] = 1
            h = np.array([flag[f] for f in funcs])
            tp, fp, fn_ = int((y & h).sum()), int(((1 - y) & h).sum()), int((y & (1 - h)).sum())
            per[c].append({"f1": 200 * tp / (2 * tp + fp + fn_), "precision": 100 * tp / max(1, tp + fp),
                           "recall": 100 * tp / max(1, int(y.sum())), "tp": tp})
    out = {"n_functions": len(funcs), "n_vul": int(y.sum()), "xfgs_outside_functions": outside, "conditions": {}}
    clean_f1 = np.array([r["f1"] for r in per["clean"]])
    for c, rows in per.items():
        f1 = np.array([r["f1"] for r in rows])
        d = f1 - clean_f1
        out["conditions"][c] = {k: float(np.mean([r[k] for r in rows])) for k in ("f1", "precision", "recall", "tp")}
        out["conditions"][c].update(f1_sd=float(f1.std(ddof=1)), delta_f1=float(d.mean()), delta_f1_sd=float(d.std(ddof=1)))
    p = ROOT / "results/stats/deepwukong_function_level.json"
    p.write_text(json.dumps(out, indent=1))
    print(p)
    for c, v in out["conditions"].items():
        print(f"{c:12s} F1 {v['f1']:.2f} R {v['recall']:.1f} flagged vul {v['tp']:.0f}/{out['n_vul']} dF1 {v['delta_f1']:+.2f}")


if __name__ == "__main__":
    main()
