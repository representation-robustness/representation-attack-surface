#!/usr/bin/env python3
"""
Generate missing JSONL files needed for 7-condition evaluation:
  - Devign compound (rename + dead + CF all at once)
  - BigVul pairwise (ren_dead, ren_cf, dead_cf) + compound
  - DiverseVul pairwise + compound

Run from anywhere:
    python ~/thesis/generate_missing_jsonl.py
"""
import json, sys, traceback
from pathlib import Path

DEVIGN_FULL = Path(__file__).resolve().parent / "devign_full"
sys.path.insert(0, str(DEVIGN_FULL))
from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow

DATASET_DIRS = {
    "devign":     Path.home() / "GNN-ReGVD/dataset_devign",
    "bigvul":     Path.home() / "GNN-ReGVD/dataset_bigvul",
    "diversevul": Path.home() / "GNN-ReGVD/dataset_diversevul",
}

# (source_split, [transforms in order], output_name)
TARGETS = {
    "devign": [
        ("test_obf_ren_dead",   "test_obf_ren_dead",   obf_controlflow),    # ren+dead → +cf
        ("test_obf_identifier", "test_obf_ren_cf",     obf_controlflow),    # ren → +cf (but ren_cf already exists)
        ("test_obf_identifier", "test_obf_dead_cf",    None),               # skip, already exists
        ("test_obf_ren_dead",   "test_obf_compound",   obf_controlflow),    # ren+dead → +cf = compound
    ],
    "bigvul": [
        ("test_obf_identifier", "test_obf_ren_dead",   obf_deadcode),
        ("test_obf_identifier", "test_obf_ren_cf",     obf_controlflow),
        ("test_obf_deadcode",   "test_obf_dead_cf",    obf_controlflow),
        ("test_obf_ren_dead",   "test_obf_compound",   obf_controlflow),
    ],
    "diversevul": [
        ("test_obf_identifier", "test_obf_ren_dead",   obf_deadcode),
        ("test_obf_identifier", "test_obf_ren_cf",     obf_controlflow),
        ("test_obf_deadcode",   "test_obf_dead_cf",    obf_controlflow),
        ("test_obf_ren_dead",   "test_obf_compound",   obf_controlflow),
    ],
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def apply_transform(rows, transform):
    out, ok, fail = [], 0, 0
    for i, row in enumerate(rows):
        try:
            new_func = transform(row["func"])
            ok += 1
        except Exception:
            new_func = row["func"]
            fail += 1
        out.append({**row, "func": new_func})
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(rows)}  ok={ok}  fail={fail}", flush=True)
    return out, ok, fail


def main():
    for dataset, targets in TARGETS.items():
        data_dir = DATASET_DIRS[dataset]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}  ({data_dir})")

        for src_name, out_name, transform in targets:
            out_path = data_dir / f"{out_name}.jsonl"

            if out_path.exists():
                print(f"  SKIP {out_name} — already exists")
                continue

            if transform is None:
                print(f"  SKIP {out_name} — no transform needed (already exists)")
                continue

            src_path = data_dir / f"{src_name}.jsonl"
            if not src_path.exists():
                print(f"  SKIP {out_name} — source {src_name}.jsonl not found")
                continue

            print(f"\n  Generating {out_name} from {src_name} + {transform.__name__}...", flush=True)
            rows = load_jsonl(src_path)
            print(f"  Loaded {len(rows)} rows", flush=True)

            out_rows, ok, fail = apply_transform(rows, transform)

            with open(out_path, "w") as f:
                for r in out_rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  Done: {ok} transformed, {fail} fallback → {out_path.name}", flush=True)

    print("\nAll missing JSONL files generated.")


if __name__ == "__main__":
    main()
