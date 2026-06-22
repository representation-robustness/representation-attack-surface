#!/usr/bin/env python3
"""
generate_pairwise_tests.py

Generate the 3 pairwise-obfuscated Devign test splits missing from the
existing single-transform and compound sets:

  test_obf_ren_dead   — identifier rename  +  dead code insertion
  test_obf_ren_cf     — identifier rename  +  control flow restructuring
  test_obf_dead_cf    — dead code insertion + control flow restructuring

Strategy: chain transforms applied to the func strings in the existing
single-transform JSONL files (already vetted for parse errors).

Output JSONL goes to ~/GNN-ReGVD/dataset_devign/ (same location as the
other test splits — all sequence models and the Joern CPG pipeline read
from there).
"""

import json, sys, traceback
from pathlib import Path

DEVIGN_FULL  = Path(__file__).resolve().parent
DATASET_DIR  = Path.home() / "GNN-ReGVD/dataset_devign"
sys.path.insert(0, str(DEVIGN_FULL))

from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow

SOURCES = {
    "test_obf_identifier":  DATASET_DIR / "test_obf_identifier.jsonl",
    "test_obf_deadcode":    DATASET_DIR / "test_obf_deadcode.jsonl",
    "test_obf_controlflow": DATASET_DIR / "test_obf_controlflow.jsonl",
}

PAIRS = {
    "test_obf_ren_dead": ("test_obf_identifier",  obf_deadcode),
    "test_obf_ren_cf":   ("test_obf_identifier",  obf_controlflow),
    "test_obf_dead_cf":  ("test_obf_deadcode",     obf_controlflow),
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def main():
    for out_name, (src_name, second_transform) in PAIRS.items():
        out_path = DATASET_DIR / f"{out_name}.jsonl"
        if out_path.exists():
            print(f"Already exists: {out_path.name}")
            continue

        print(f"\nGenerating {out_name}...", flush=True)
        rows = load_jsonl(SOURCES[src_name])
        print(f"  Loaded {len(rows)} rows from {src_name}", flush=True)

        out_rows = []
        ok = fail = 0
        for i, row in enumerate(rows):
            try:
                new_func = second_transform(row["func"])
                ok += 1
            except Exception:
                new_func = row["func"]   # fallback to first-transform version
                fail += 1
            out_rows.append({**row, "func": new_func})

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(rows)}  ok={ok}  fail={fail}", flush=True)

        with open(out_path, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

        print(f"  Done: {ok} transformed, {fail} fallback → {out_path}", flush=True)

    print("\nAll pairwise splits generated.")


if __name__ == "__main__":
    main()
