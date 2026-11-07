"""Recompute every derived result from the prediction files that exist right now.

  python -m pipeline.report.refresh

metrics (per model) -> stats (per dataset) -> attacks/transfer (main detectors) -> tables -> analyses.
Safe to run repeatedly while training is in progress; outputs are overwritten.
"""
from pipeline.eval import attacks, metrics, stats
from pipeline.models.common import ROOT
from pipeline.report import analyses, tables

ATTACK_MODELS = ["ecg_rgcn", "ecg_rgcn_struct", "angle", "angle_struct", "vulgnn", "vulgnn_struct",
                 "reveal", "reveal_struct", "regvd", "lmggnn", "codebert", "codet5plus", "tfidf_lr", "cpg_lr"]


def main():
    for ds in tables.DATASETS:
        pdir = ROOT / "results" / "preds" / ds
        if not pdir.exists():
            continue
        models = sorted(p.name for p in pdir.iterdir() if any(p.glob("run_*.json.gz")))
        for m in models:
            metrics.summarize(ds, m)
        _stats(ds)
        present = [m for m in ATTACK_MODELS if m in models]
        out = ROOT / "results" / "attacks" / ds
        out.mkdir(parents=True, exist_ok=True)
        import json
        for m in present:
            (out / f"minbudget_{m}.json").write_text(json.dumps(attacks.detector_attacks(ds, m), indent=1))
        (out / "transfer.json").write_text(json.dumps(
            [attacks.transfer(ds, s, t) for s in present for t in present if s != t], indent=1))
        print(f"{ds}: {len(models)} models refreshed ({', '.join(models)})")
    tables.main()
    analyses.main()


def _stats(ds):
    import csv
    import json
    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    for scope in ("valid", "all"):
        rows = stats.table(ds, scope)
        (out / f"{ds}_{scope}.json").write_text(json.dumps(rows, indent=1))
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{ds}_{scope}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
