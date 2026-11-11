"""Draw the manual semantic-audit sample (seeded) and write it as a review sheet.

  python -m pipeline.transforms.audit_sample [--per-cell 10] [--seed 2026]

For each dataset x single transformation family, samples validly transformed test
functions and writes unified diffs to $RERUN_ROOT/results/audit/audit_sample.md with a
checkbox per item. Reviewers mark each diff as semantics-preserving or not; the
filled-in sheet is the evidence for the manual-audit rate reported in the paper.
"""
import argparse
import difflib
import json
import random

from pipeline.models.common import ROOT, load_text

FAMILIES = ["identifier", "deadcode", "controlflow"]
DATASETS = ["devign", "diversevul", "bigvul", "primevul"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = ROOT / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Manual semantic audit sample", "",
             f"Seed {args.seed}; {args.per_cell} validly transformed test functions per dataset x family.",
             "Mark each item: [x] semantics preserved, or [ ] and write the problem under it.", ""]
    index = []
    for ds in args.datasets:
        if not (ROOT / "data" / ds / "clean/test.jsonl").exists():
            continue
        clean = {r["id"]: r["func"] for r in load_text(ds, "test")}
        for fam in FAMILIES:
            valid = [json.loads(l)["id"] for l in open(ROOT / "data" / ds / "validity" / "test" / f"{fam}.jsonl")
                     if json.loads(l)["valid"]]
            trans = {r["id"]: r["func"] for r in load_text(ds, "test", fam)}
            for sid in sorted(rng.sample(valid, min(args.per_cell, len(valid)))):
                diff = difflib.unified_diff(clean[sid].splitlines(), trans[sid].splitlines(),
                                            "original", fam, lineterm="", n=2)
                lines += [f"## {ds} / {fam} / {sid}", "", "- [ ] semantics preserved", "", "```diff", *diff, "```", ""]
                index.append({"dataset": ds, "family": fam, "id": sid})
    (out / "audit_sample.md").write_text("\n".join(lines))
    (out / "audit_index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} items -> {out / 'audit_sample.md'}")


if __name__ == "__main__":
    main()
