"""Attack analyses computed from saved predictions (no model queries happen here).

Per run (seed) r of a detector f, with the validation threshold of that run:
  X_f  = vulnerable test functions that f classifies correctly when clean.
  A function counts as evaded by condition c only if c's transformation of it is VALID
  (validity gate) and f(c(x)) = 0.

1. Fixed-policy attack (realisable under the no-query threat model): the attacker always
   applies one condition c. ASR_fixed(c) = |evaded by c| / |X_f restricted to valid c|.
2. Minimum-budget audit (offline ORACLE analysis, uses f's predictions to pick the
   sequence; reported as worst-case sensitivity, not an attack): budget k = number of
   transformation families. ASR@k = |{x in X_f : some valid condition with <= k families
   evades f}| / |X_f|. Ties at the minimal budget are broken by fewest added tokens, then
   by the fixed order of CONDITIONS. Reported with the edit cost of the chosen sequence.
3. Transfer: for source s and target t (runs paired by seed when both share the seeds, otherwise
   positionally: i-th neural seed with i-th non-neural subsample run), on X_{s,t} = vulnerable functions
   both classify correctly, the source's minimal sequence A(x; s) is applied; TASR =
   |{x : s evaded and t(A(x;s)) = 0}| / |{x : s evaded}|. Denominators are reported.

  python -m pipeline.eval.attacks --dataset devign --models reveal,codebert,...
"""
import argparse
import gzip
import json
from collections import defaultdict
from functools import lru_cache

import numpy as np

from pipeline.models.common import CONDITIONS, ROOT

FAMILIES = {"identifier": 1, "deadcode": 1, "controlflow": 1, "ren_dead": 2, "ren_cf": 2, "dead_cf": 2, "compound": 3}


@lru_cache(maxsize=None)
def validity(dataset: str, cond: str) -> dict:
    """id -> (valid, edit cost dict) for one condition."""
    out = {}
    for line in open(ROOT / "data" / dataset / "validity" / "test" / f"{cond}.jsonl"):
        r = json.loads(line)
        cost = {"identifiers_renamed": 0, "occurrences_renamed": 0, "sites": 0, "tokens_added": 0}
        for t in r.get("transforms", []):
            if t.get("transform") == "identifier":
                cost["identifiers_renamed"] += t.get("n_names", 0)
                cost["occurrences_renamed"] += t.get("n_occurrences", 0)
            else:
                cost["sites"] += t.get("n_sites", 0)
                cost["tokens_added"] += t.get("tokens_added", 0)
        out[r["id"]] = (r["valid"], cost)
    return out


def load_run(dataset: str, model: str, run: int) -> dict:
    with gzip.open(ROOT / "results" / "preds" / dataset / model / f"run_{run}.json.gz", "rt") as f:
        r = json.load(f)
    thr = r["threshold"]
    pred = {c: {i: int(p >= thr) for i, p in zip(v["ids"], v["p"])} for c, v in r["test"].items()}
    label = dict(zip(r["test"]["clean"]["ids"], r["test"]["clean"]["y"]))
    return {"pred": pred, "label": label}


def runs_of(dataset: str, model: str) -> list[int]:
    return sorted(int(p.name[4:-8]) for p in (ROOT / "results" / "preds" / dataset / model).glob("run_*.json.gz"))


def paired_runs(dataset: str, a: str, b: str) -> list[tuple[int, int]]:
    ra, rb = runs_of(dataset, a), runs_of(dataset, b)
    if set(ra) == set(rb):
        return [(r, r) for r in ra]
    return list(zip(ra, rb))


def minimal_sequence(dataset, run, x):
    """Lowest-budget valid condition that evades run for x, tie -> fewest tokens added."""
    best = None
    for c in CONDITIONS:
        ok, cost = validity(dataset, c).get(x, (False, None))
        if not ok or run["pred"][c].get(x, 1) != 0:
            continue
        key = (FAMILIES[c], cost["tokens_added"], CONDITIONS.index(c))
        if best is None or key < best[0]:
            best = (key, c, cost)
    return best


def detector_attacks(dataset: str, model: str) -> dict:
    per_run = []
    for r in runs_of(dataset, model):
        run = load_run(dataset, model, r)
        X = [x for x, y in run["label"].items() if y == 1 and run["pred"]["clean"].get(x) == 1]
        fixed = {}
        for c in CONDITIONS:
            Xc = [x for x in X if validity(dataset, c).get(x, (False,))[0] and x in run["pred"][c]]
            ev = sum(run["pred"][c][x] == 0 for x in Xc)
            fixed[c] = {"asr": 100 * ev / max(1, len(Xc)), "evaded": ev, "denominator": len(Xc)}
        chosen = {x: minimal_sequence(dataset, run, x) for x in X}
        asr_k = {k: 100 * sum(1 for v in chosen.values() if v and v[0][0] <= k) / max(1, len(X)) for k in (1, 2, 3)}
        evaded = [v for v in chosen.values() if v]
        costs = defaultdict(list)
        for v in evaded:
            for k2, val in v[2].items():
                costs[k2].append(val)
        per_run.append({"run": r, "denominator": len(X), "fixed_policy": fixed, "asr_at_k": asr_k,
                        "mean_min_budget": float(np.mean([v[0][0] for v in evaded])) if evaded else None,
                        "mean_cost_of_min_sequence": {k2: float(np.mean(v)) for k2, v in costs.items()}})
    return {"dataset": dataset, "model": model, "per_run": per_run,
            "asr_at_k_mean": {k: float(np.mean([p["asr_at_k"][k] for p in per_run])) for k in (1, 2, 3)},
            "asr_at_k_sd": {k: float(np.std([p["asr_at_k"][k] for p in per_run], ddof=1)) if len(per_run) > 1 else 0.0
                            for k in (1, 2, 3)},
            "fixed_policy_mean": {c: float(np.mean([p["fixed_policy"][c]["asr"] for p in per_run])) for c in CONDITIONS}}


def transfer(dataset: str, src: str, tgt: str) -> dict:
    rows = []
    for r, r2 in paired_runs(dataset, src, tgt):
        s, t = load_run(dataset, src, r), load_run(dataset, tgt, r2)
        X = [x for x, y in s["label"].items()
             if y == 1 and s["pred"]["clean"].get(x) == 1 and t["pred"]["clean"].get(x) == 1]
        succ = [(x, m[1]) for x in X if (m := minimal_sequence(dataset, s, x))]
        hit = sum(1 for x, c in succ if t["pred"][c].get(x, 1) == 0)
        rows.append({"run": r, "target_run": r2, "joint_correct": len(X), "source_evaded": len(succ), "transferred": hit,
                     "tasr": 100 * hit / max(1, len(succ))})
    return {"source": src, "target": tgt, "per_run": rows,
            "tasr_mean": float(np.mean([x["tasr"] for x in rows])) if rows else None,
            "tasr_sd": float(np.std([x["tasr"] for x in rows], ddof=1)) if len(rows) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", required=True, help="comma list; transfer is computed for every ordered pair")
    args = ap.parse_args()
    models = args.models.split(",")
    out = ROOT / "results" / "attacks" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for m in models:
        res = detector_attacks(args.dataset, m)
        (out / f"minbudget_{m}.json").write_text(json.dumps(res, indent=1))
        print(m, {k: round(v, 1) for k, v in res["asr_at_k_mean"].items()})
    tr = [transfer(args.dataset, s, t) for s in models for t in models if s != t]
    (out / "transfer.json").write_text(json.dumps(tr, indent=1))
    for x in tr:
        print(f"  {x['source']:>12s} -> {x['target']:<12s} TASR={x['tasr_mean']:.1f}")


if __name__ == "__main__":
    main()
