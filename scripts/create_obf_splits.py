#!/usr/bin/env python3
"""Generate all 7 obfuscated test split variants for BigVul or DiverseVul.

Reads test.jsonl from the given splits directory and writes seven obfuscated
variants alongside it. Each record must have a 'func' field containing C source.

Usage:
    python scripts/create_obf_splits.py <splits_dir>

Examples:
    python scripts/create_obf_splits.py bigvul/splits/
    python scripts/create_obf_splits.py diversevul_dataset/splits/

Output files written into <splits_dir>/:
    test_obf_identifier.jsonl   — identifier renaming
    test_obf_deadcode.jsonl     — dead-code insertion
    test_obf_controlflow.jsonl  — control-flow restructuring
    test_obf_ren_dead.jsonl     — renaming + dead-code
    test_obf_ren_cf.jsonl       — renaming + control-flow
    test_obf_dead_cf.jsonl      — dead-code + control-flow
    test_obf_compound.jsonl     — all three combined

Requires: tree-sitter==0.25.2, tree-sitter-c==0.24.1 (see requirements.txt)
"""
import json
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS))
from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow

CONDITIONS = {
    "test_obf_identifier":  [obf_identifier],
    "test_obf_deadcode":    [obf_deadcode],
    "test_obf_controlflow": [obf_controlflow],
    "test_obf_ren_dead":    [obf_identifier, obf_deadcode],
    "test_obf_ren_cf":      [obf_identifier, obf_controlflow],
    "test_obf_dead_cf":     [obf_deadcode,   obf_controlflow],
    "test_obf_compound":    [obf_identifier, obf_deadcode, obf_controlflow],
}


def apply_transforms(code: str, transforms) -> str:
    for fn in transforms:
        try:
            code = fn(code)
        except Exception:
            pass  # keep original on parse/transform failure
    return code


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    splits_dir = Path(sys.argv[1])
    src = splits_dir / "test.jsonl"
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records):,} records from {src}")

    for name, transforms in CONDITIONS.items():
        out_path = splits_dir / f"{name}.jsonl"
        label = "+".join(f.__name__.replace("obf_", "") for f in transforms)
        print(f"  {name}.jsonl ({label}) ...", end=" ", flush=True)

        errors = 0
        lines = []
        for rec in records:
            new_rec = dict(rec)
            original = rec["func"]
            new_rec["func"] = apply_transforms(original, transforms)
            if new_rec["func"] == original:
                errors += 1
            lines.append(json.dumps(new_rec))

        out_path.write_text("\n".join(lines) + "\n")
        note = f"  ({errors} kept original)" if errors else ""
        print(f"done{note}")

    print(f"\nAll 7 conditions written to {splits_dir}/")


if __name__ == "__main__":
    main()
