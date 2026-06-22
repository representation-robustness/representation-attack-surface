#!/usr/bin/env python3
"""
generate_compound_test.py — Create compound-obfuscated test data for Devign.

Applies identifier renaming → dead code insertion → control flow restructuring
in sequence to produce a single compound test condition.

Outputs:
  devign_full/obf_compound/              — compound C files (for GNN models)
  devign_full/obf_compound_data.json     — compound JSON (for sequence models)
"""
import json, sys
from pathlib import Path

DEVIGN_FULL = Path(__file__).resolve().parent
THESIS_ROOT = DEVIGN_FULL.parent
sys.path.insert(0, str(DEVIGN_FULL))

from obf_transforms_v2 import obf_identifier, obf_deadcode, obf_controlflow

ORIGINALS_DIR = DEVIGN_FULL / "originals"
COMPOUND_DIR  = DEVIGN_FULL / "obf_compound"
SPLIT_FILE    = DEVIGN_FULL / "devign_full_split_801010.json"
ORIG_JSON     = DEVIGN_FULL / "originals_full_data_with_slices.json"
OUT_JSON      = DEVIGN_FULL / "obf_compound_data.json"

COMPOUND_DIR.mkdir(exist_ok=True)


def apply_compound(src: str) -> str:
    s = obf_identifier(src)
    s = obf_deadcode(s)
    s = obf_controlflow(s)
    return s


def main():
    split = json.loads(SPLIT_FILE.read_text())
    test_files = set(split["splits"]["test"])
    print(f"Test files: {len(test_files)}")

    # ── Generate compound C files (for GNN models) ────────────────────────────
    c_ok, c_fail = 0, 0
    for fname in sorted(test_files):
        src_path = ORIGINALS_DIR / fname
        dst_path = COMPOUND_DIR / fname
        if dst_path.exists():
            c_ok += 1
            continue
        if not src_path.exists():
            c_fail += 1
            continue
        try:
            src = src_path.read_text(encoding="utf-8", errors="replace")
            dst_path.write_text(apply_compound(src), encoding="utf-8")
            c_ok += 1
        except Exception as e:
            # fallback to original
            import shutil
            shutil.copy2(src_path, dst_path)
            c_fail += 1

    print(f"C files: {c_ok} ok, {c_fail} failed/fallback")

    # ── Generate compound JSON (for sequence models: ReGVD, CodeBERT, TF-IDF) ─
    if OUT_JSON.exists():
        print(f"Compound JSON already exists: {OUT_JSON}")
        return

    orig_data = json.loads(ORIG_JSON.read_text())
    orig_idx  = {r["file_name"]: r for r in orig_data}

    compound_records = []
    j_ok, j_fail = 0, 0
    for fname in sorted(test_files):
        if fname not in orig_idx:
            continue
        rec = orig_idx[fname]
        try:
            new_code = apply_compound(rec["code"])
            j_ok += 1
        except Exception:
            new_code = rec["code"]
            j_fail += 1
        compound_records.append({
            "file_name": rec["file_name"],
            "code":      new_code,
            "label":     rec["label"],
        })

    OUT_JSON.write_text(json.dumps(compound_records))
    print(f"JSON: {len(compound_records)} records ({j_ok} transformed, {j_fail} fallback)")
    print(f"Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
