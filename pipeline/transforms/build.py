"""Build transformed splits and their validity records.

  python -m pipeline.transforms.build --dataset devign --split test [--limit N] [--workers 8]

Outputs under $RERUN_ROOT/data/<dataset>/:
  transformed/<split>/<condition>.jsonl   {"id", "func", "target"}  (all samples, same order as clean)
  validity/<split>/<condition>.jsonl      per-sample validity record + transform info
  validity/<split>/summary.json           per-condition rates and reject reasons
Every sample is kept in the transformed files; analyses select the valid subset by id.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from pipeline.transforms.core import CONDITIONS, apply_condition, parse
from pipeline.transforms.validate import check

ROOT = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun"))


def process(rec: dict) -> dict:
    lang = parse(rec["func"]).lang
    cache: dict = {}
    out = {}
    for cond in CONDITIONS:
        new, infos = apply_condition(rec["func"], cond)
        inv = {}
        for info in infos:
            inv.update({f: o for o, f in (info.pop("mapping", None) or {}).items()})
        v = check(rec["func"], new, lang, cache, rename_inverse=inv)
        v["transforms"] = infos
        out[cond] = (new, v)
    return {"id": rec["id"], "target": rec["target"], "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for pilot outputs")
    args = ap.parse_args()

    base = ROOT / "data" / args.dataset
    recs = [json.loads(l) for l in open(base / "clean" / f"{args.split}.jsonl")]
    if args.limit:
        recs = recs[:args.limit]
    tdir = base / f"transformed{args.tag}" / args.split
    vdir = base / f"validity{args.tag}" / args.split
    tdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    tf = {c: open(tdir / f"{c}.jsonl", "w") for c in CONDITIONS}
    vf = {c: open(vdir / f"{c}.jsonl", "w") for c in CONDITIONS}

    stats = {c: Counter() for c in CONDITIONS}
    sites = {c: defaultdict(list) for c in CONDITIONS}
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=500) as pool:
        for i, r in enumerate(pool.imap(process, recs, chunksize=8)):  # imap keeps clean order
            for c, (new, v) in r["out"].items():
                tf[c].write(json.dumps({"id": r["id"], "func": new, "target": r["target"]}) + "\n")
                vf[c].write(json.dumps({"id": r["id"], **v}) + "\n")
                s = stats[c]
                s["n"] += 1
                s["func_found"] += v["func_found"]
                s["changed"] += v["changed"]
                s["valid"] += v["valid"]
                s["lang_" + v["lang"]] += 1
                if not v["valid"]:
                    s["reject_" + v["reject_reason"]] += 1
                for t in v["transforms"]:
                    for k in ("n_sites", "n_occurrences", "tokens_added"):
                        if k in t:
                            sites[c][f"{t['transform']}.{k}"].append(t[k])
            if (i + 1) % 1000 == 0:
                print(f"[{i + 1}/{len(recs)}] {time.time() - t0:.0f}s", flush=True)
    for f in [*tf.values(), *vf.values()]:
        f.close()

    summary = {"dataset": args.dataset, "split": args.split, "n": len(recs),
               "seconds": round(time.time() - t0, 1), "conditions": {}}
    for c, s in stats.items():
        n = s["n"] or 1
        summary["conditions"][c] = {
            **dict(s),
            "rate_func_found": round(s["func_found"] / n, 4),
            "rate_changed": round(s["changed"] / n, 4),
            "rate_valid": round(s["valid"] / n, 4),
            "rate_valid_given_changed": round(s["valid"] / max(1, s["changed"]), 4),
            "mean_edits": {k: round(sum(v) / len(v), 2) for k, v in sites[c].items() if v},
        }
    (vdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({c: {k: v for k, v in d.items() if k.startswith("rate") or k.startswith("reject")}
                      for c, d in summary["conditions"].items()}, indent=1))
    print(f"done in {summary['seconds']}s")


if __name__ == "__main__":
    main()
