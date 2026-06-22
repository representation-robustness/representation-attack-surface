#!/usr/bin/env python3
"""Generate pairwise obfuscated JSON files from originals_full_data_with_slices.json.

Produces:
  obf_ren_dead_full_data_with_slices.json   (identifier + deadcode)
  obf_ren_cf_full_data_with_slices.json     (identifier + controlflow)
  obf_dead_cf_full_data_with_slices.json    (deadcode + controlflow)
  obf_compound_full_data_with_slices.json   (all three)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow

DEVIGN_FULL = Path(__file__).resolve().parent
ORIG_FILE   = DEVIGN_FULL / "originals_full_data_with_slices.json"

TASKS = {
    "obf_ren_dead_full_data_with_slices.json":   [obf_identifier, obf_deadcode],
    "obf_ren_cf_full_data_with_slices.json":     [obf_identifier, obf_controlflow],
    "obf_dead_cf_full_data_with_slices.json":    [obf_deadcode,   obf_controlflow],
    "obf_compound_full_data_with_slices.json":   [obf_identifier, obf_deadcode, obf_controlflow],
}

print("Loading originals...", flush=True)
orig = json.load(open(ORIG_FILE))
print(f"  {len(orig)} records", flush=True)

for out_name, fns in TASKS.items():
    out_path = DEVIGN_FULL / out_name
    label = "+".join(f.__name__.replace("obf_","") for f in fns)
    print(f"\nGenerating {out_name} ({label})...", flush=True)

    results = []
    errors  = 0
    for i, rec in enumerate(orig):
        code = rec["code"]
        try:
            for fn in fns:
                code = fn(code)
        except Exception as e:
            errors += 1
            code = rec["code"]  # fall back to original on error

        new_rec = dict(rec)
        new_rec["code"] = code
        results.append(new_rec)

        if (i + 1) % 5000 == 0 or (i + 1) == len(orig):
            print(f"  [{i+1}/{len(orig)}] errors={errors}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"  Saved → {out_path}  ({errors} errors)", flush=True)

print("\nDone.", flush=True)
