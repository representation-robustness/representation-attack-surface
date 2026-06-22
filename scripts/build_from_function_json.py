#!/usr/bin/env python3
"""Extract all functions from ReVeal function.json into devign_full/* variants."""
import csv
import json
import re
from pathlib import Path

from obf_transforms import (
    build_mapping,
    insert_deadcode,
    rewrite_controlflow,
    transform_identifiers,
)

THESIS_ROOT = Path(__file__).resolve().parent.parent
SRC_JSON = THESIS_ROOT / "data/raw/ReVeal/data/function.json"
BASE = THESIS_ROOT / "devign_full"

DIRS = {
    "originals": BASE / "originals",
    "obf_identifier": BASE / "obf_identifier",
    "obf_deadcode": BASE / "obf_deadcode",
    "obf_controlflow": BASE / "obf_controlflow",
}

MANIFEST = BASE / "originals_manifest.csv"


def slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def main():
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

    with SRC_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, row in enumerate(data, start=1):
        fname = f"{idx:05d}_{slug(str(row.get('project', 'unknown')))}_{str(row.get('commit_id', 'unknown'))[:8].lower()}.c"
        func = row.get("func", "")
        text = func if func.endswith("\n") else func + "\n"
        (DIRS["originals"] / fname).write_text(text, encoding="utf-8")
        rows.append({
            "index": idx,
            "filename": fname,
            "project": str(row.get("project", "")),
            "project_slug": slug(str(row.get("project", ""))),
            "commit_id": str(row.get("commit_id", "")),
            "commit8": str(row.get("commit_id", ""))[:8].lower(),
            "target": int(row["target"]),
            "source_json_index": idx - 1,
        })

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "index", "filename", "project", "project_slug",
                "commit_id", "commit8", "target", "source_json_index",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"wrote {n} originals to {DIRS['originals']}")
    print(f"manifest: {MANIFEST}")

    id_changed = 0
    cf_changed = 0
    for r in rows:
        path = DIRS["originals"] / r["filename"]
        code = path.read_text(encoding="utf-8")
        tag = r["index"]
        obf_id = transform_identifiers(code, build_mapping(code))
        if obf_id != code:
            id_changed += 1
        (DIRS["obf_identifier"] / r["filename"]).write_text(obf_id, encoding="utf-8")
        (DIRS["obf_deadcode"] / r["filename"]).write_text(insert_deadcode(code, tag), encoding="utf-8")
        obf_cf = rewrite_controlflow(code, tag)
        if obf_cf != code:
            cf_changed += 1
        (DIRS["obf_controlflow"] / r["filename"]).write_text(obf_cf, encoding="utf-8")
    print(f"obf_identifier: {n} files, {id_changed} content-changed")
    print(f"obf_deadcode: {n} files")
    print(f"obf_controlflow: {n} files, {cf_changed} body-rewritten (rest fallback)")

    t0 = sum(1 for r in rows if r["target"] == 0)
    t1 = sum(1 for r in rows if r["target"] == 1)
    print(f"labels: target0={t0}, target1={t1}")


if __name__ == "__main__":
    main()
