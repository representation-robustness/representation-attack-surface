"""Compare a rerun against a reference run, artifact by artifact, keyed by sample id.

  python -m pipeline.report.compare --ref /path/to/reference/rerun [--new $RERUN_ROOT] [--tol 1e-6]

Checks whatever exists in BOTH roots:
  data/<ds>/clean/*.jsonl, transformed/**, validity/**   records must be identical per id
  data/<ds>/graphs/<split>/<cond>.jsonl.gz               node/edge lists identical per id after relabelling
                                                         Joern's run-specific node ids (ids present in both)
  results/preds/<ds>/<model>/run_*.json.gz               max |p_new - p_ref| per run, threshold delta
  results/metrics/<ds>/<model>.json                      F1 deltas per condition
Prints one line per artifact and exits 1 if any deterministic artifact differs or a CPU model's
predictions differ by more than --tol. Neural models are reported (GPU kernels are not bit-exact), not failed.
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

CPU_MODELS = {"tfidf_lr", "cpg_lr"}


def _records(path: Path) -> dict:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def _canonical(r: dict) -> dict:
    """Joern node ids are global to a CPG and vary between runs; relabel them by rank within the record."""
    if "nodes" not in r:
        return r
    order = {n[0]: i for i, n in enumerate(sorted(r["nodes"], key=lambda n: n[0]))}
    return {**r, "nodes": sorted([order[n[0]], *n[1:]] for n in r["nodes"]),
            "edges": sorted([order[s], order[d], t] for s, d, t in r["edges"])}


def compare_records(ref: Path, new: Path, first_pass_only=False) -> tuple[bool, str]:
    a, b = _records(ref), _records(new)
    if first_pass_only:  # header-normalised retries are batched differently in a --limit slice
        b = {i: r for i, r in b.items() if not r.get("header_normalized")}
    common = a.keys() & b.keys()
    diff = [i for i in common if _canonical(a[i]) != _canonical(b[i])]
    # graphs may be built for a subset (--limit), but never an empty one
    ok = not diff and (len(common) == len(a) == len(b) or ("graphs" in ref.parts and len(common) == len(b) > 0))
    note = f"{len(common)} common ids ({len(a)} ref / {len(b)} new), {len(diff)} differ"
    return ok, note + (f", e.g. {sorted(diff)[:3]}" if diff else "")


def compare_preds(ref: Path, new: Path) -> tuple[float, float]:
    a, b = json.load(gzip.open(ref)), json.load(gzip.open(new))
    worst = 0.0
    for cond, ra in a["test"].items():
        rb = b["test"].get(cond)
        if rb is None:
            continue
        pb = dict(zip(rb["ids"], rb["p"]))
        worst = max([worst] + [abs(p - pb[i]) for i, p in zip(ra["ids"], ra["p"]) if i in pb])
    return worst, abs(a["threshold"] - b["threshold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--new", type=Path, default=Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun")))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--graph-slice", action="store_true",
                    help="new graphs were built with --limit: compare first-pass clean graphs only")
    a = ap.parse_args()
    failed = False

    for kind, pattern in [("clean", "data/*/clean/*.jsonl"), ("transformed", "data/*/transformed/*/*.jsonl"),
                          ("validity", "data/*/validity/*/*.jsonl"), ("graphs", "data/*/graphs/*/*.jsonl.gz")]:
        for new in sorted(a.new.glob(pattern)):
            ref = a.ref / new.relative_to(a.new)
            if not ref.exists() or (kind == "graphs" and a.graph_slice and new.name != "clean.jsonl.gz"):
                continue
            ok, note = compare_records(ref, new, first_pass_only=kind == "graphs" and a.graph_slice)
            failed |= not ok
            print(f"{'OK  ' if ok else 'DIFF'} {kind:11s} {new.relative_to(a.new)}: {note}")

    for new in sorted(a.new.glob("results/preds/*/*/run_*.json.gz")):
        ref = a.ref / new.relative_to(a.new)
        if not ref.exists():
            continue
        model = new.parent.name
        dp, dt = compare_preds(ref, new)
        ok = dp <= a.tol and dt <= a.tol
        strict = model in CPU_MODELS
        failed |= strict and not ok
        tag = "OK  " if ok else ("DIFF" if strict else "INFO")
        print(f"{tag} preds       {new.relative_to(a.new)}: max|dp|={dp:.2e} d_threshold={dt:.2e}")

    for new in sorted(a.new.glob("results/metrics/*/*.json")):
        ref = a.ref / new.relative_to(a.new)
        if ref.exists():
            print(f"INFO metrics     {new.relative_to(a.new)}: {_metric_delta(ref, new)}")
    print("RESULT:", "DIFFERENCES FOUND" if failed else "all deterministic artifacts identical")
    sys.exit(1 if failed else 0)


def _metric_delta(ref: Path, new: Path) -> str:
    """Largest change, over conditions, in mean clean F1 and mean Delta-F1 (validly transformed scope)."""
    a, b = json.loads(ref.read_text())["valid"], json.loads(new.read_text())["valid"]
    common = a.keys() & b.keys()
    if not common:
        return "no common conditions"
    dc = max(abs(a[c]["f1"]["clean_mean"] - b[c]["f1"]["clean_mean"]) for c in common)
    dd = max(abs(a[c]["f1"]["delta_mean"] - b[c]["f1"]["delta_mean"]) for c in common)
    return f"max |d clean F1|={dc:.4f}, max |d Delta-F1|={dd:.4f} over {len(common)} conditions"


if __name__ == "__main__":
    main()
