#!/usr/bin/env python3
"""
Generate new attack transformation test sets for vocab_shift, benign_token, and tempvar.

For each transform:
  1. Apply to all test-set functions (using split from devign_full_split_801010.json)
  2. Write obf_<transform>_full_data_with_slices.json (raw source for text models)
  3. Write individual .c files to devign_full/obf_<transform>/ (for Joern parsing later)

Token-aware models (TF-IDF, CodeBERT, CodeT5+, ReGVD) only need the JSON file.
Graph models (REVEAL, ECG RGCN, VulGNN, ANGLE) need Joern parsing + GGNN extraction
  -- that step is handled by a separate script.
"""

import json, sys, traceback
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_ROOT / "utils"))

from obf_transforms_v2 import obf_vocab_shift, obf_benign_token, obf_tempvar

DEVIGN_FULL = THESIS_ROOT / "devign_full"
SPLIT_FILE  = DEVIGN_FULL / "devign_full_split_801010.json"
ORIGINALS   = DEVIGN_FULL / "originals"

TRANSFORMS = {
    "vocab_shift":  obf_vocab_shift,
    "benign_token": obf_benign_token,
    "tempvar":      obf_tempvar,
}

def main():
    split = json.loads(SPLIT_FILE.read_text())
    test_names  = set(split["splits"]["test"])
    total_names = set(split["splits"]["train"]) | set(split["splits"]["valid"]) | test_names

    # Load original full_data_with_slices.json (keyed by file_name)
    print("Loading originals_full_data_with_slices.json ...", flush=True)
    orig_recs = json.loads((DEVIGN_FULL / "originals_full_data_with_slices.json").read_text())
    orig_map = {r["file_name"]: r for r in orig_recs}
    print(f"  {len(orig_map)} total records, {len(test_names)} test names", flush=True)

    for transform_name, transform_fn in TRANSFORMS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {transform_name} ...", flush=True)

        out_c_dir  = DEVIGN_FULL / f"obf_{transform_name}"
        out_c_dir.mkdir(exist_ok=True)

        out_json = DEVIGN_FULL / f"obf_{transform_name}_full_data_with_slices.json"

        new_recs = []
        ok = changed = errors = 0

        for name in sorted(orig_map.keys()):
            rec = orig_map[name]
            src = rec["code"]
            try:
                new_src = transform_fn(src)
                ok += 1
                if new_src != src:
                    changed += 1
            except Exception as e:
                new_src = src  # fall back to original on error
                errors += 1
                if errors <= 5:
                    print(f"  ERROR {name}: {e}", flush=True)

            new_rec = dict(rec)
            new_rec["code"] = new_src
            new_rec["transform"] = transform_name
            new_recs.append(new_rec)

            # Also write individual .c file (for Joern parsing if needed)
            if name in test_names:
                (out_c_dir / name).write_text(new_src, encoding="utf-8", errors="replace")

        out_json.write_text(json.dumps(new_recs, indent=None, separators=(",", ":")))
        print(f"  ok={ok}, changed={changed}, errors={errors}", flush=True)
        print(f"  JSON → {out_json}", flush=True)
        print(f"  C files → {out_c_dir}/ ({len(list(out_c_dir.glob('*.c')))} test files)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
