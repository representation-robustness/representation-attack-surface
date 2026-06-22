#!/usr/bin/env python3
"""
Exp 7a: REVEAL-Aug data preparation.

Parses obf_deadcode and obf_controlflow TRAINING split functions through
Joern to generate nodes.csv/edges.csv, then builds GGNN input JSONs for
augmented REVEAL training.

REVEAL's 169-dim node features = 69-dim structural one-hot + 100-dim Word2Vec.
  - obf_identifier: structural topology identical to originals → SKIP
  - obf_deadcode:   adds new CPG nodes/edges → parse + build
  - obf_controlflow: changes graph topology → parse + build

Outputs:
  - devign_input/parsed_cache/obf_deadcode/    (train split Joern output)
  - devign_input/parsed_cache/obf_controlflow/ (train split Joern output)
  - devign_input/obf_deadcode_train/train_GGNNinput.json
  - devign_input/obf_controlflow_train/train_GGNNinput.json

Runtime: ~10 minutes (Joern batch-parses ~2500 files in ~35 seconds)
"""

import importlib.util, json, shutil, subprocess
from pathlib import Path

THESIS_ROOT     = Path(__file__).resolve().parents[1]
BASE            = THESIS_ROOT / "devign_full"
CODE_SLICER_DIR = THESIS_ROOT / "data/raw/ReVeal/code-slicer"
CREATE_GGNN_PATH= THESIS_ROOT / "data/raw/ReVeal/data_processing/create_ggnn_data.py"
DEVIGN_INPUT    = BASE / "devign_input"
PARSED_CACHE    = DEVIGN_INPUT / "parsed_cache"

SPLIT_FILE = BASE / "devign_full_split_801010.json"
VARIANTS   = ["obf_deadcode", "obf_controlflow"]
BATCH_SIZE = 2500


# ---------------------------------------------------------------------------
# Load modules and data
# ---------------------------------------------------------------------------

def load_create_ggnn_module():
    spec = importlib.util.spec_from_file_location("create_ggnn_data", str(CREATE_GGNN_PATH))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_graph_json(mod, model, parsed_root: Path, names, labels):
    rows, skipped = [], []
    for fname in names:
        node_csv = parsed_root / fname / "nodes.csv"
        edge_csv = parsed_root / fname / "edges.csv"
        if not node_csv.exists() or not edge_csv.exists():
            skipped.append(fname)
            continue
        gi = mod.inputGeneration(
            str(node_csv), str(edge_csv),
            int(labels[fname]),
            model, mod.edgeType_full, False,
        )
        if gi is None:
            skipped.append(fname)
            continue
        rows.append(gi)
    return rows, skipped


# ---------------------------------------------------------------------------
# Step 1: Parse variant files through Joern
# ---------------------------------------------------------------------------

def parse_variant_train(variant: str, train_names: list):
    """Batch-parse obfuscated C files through joern-parse.

    Old Joern v0.3 fails if parsed/ already exists in the working directory.
    Fix: run each batch in a fresh /tmp working dir.
    """
    src_dir        = BASE / variant
    parsed_cache_v = PARSED_CACHE / variant
    parsed_cache_v.mkdir(parents=True, exist_ok=True)

    JOERN_PARSE = CODE_SLICER_DIR / "joern" / "joern-parse"

    # Skip files already parsed (allow resuming)
    to_parse = [f for f in train_names
                if not (parsed_cache_v / f / "nodes.csv").exists()]
    print(f"  {variant}: {len(to_parse)} of {len(train_names)} need parsing", flush=True)
    if not to_parse:
        print(f"  {variant}: all train files already parsed — skipping", flush=True)
        return

    for batch_idx, off in enumerate(range(0, len(to_parse), BATCH_SIZE)):
        batch = to_parse[off: off + BATCH_SIZE]
        pct   = (off + len(batch)) / len(to_parse) * 100
        print(f"  Parsing batch {batch_idx + 1}: {len(batch)} files ({pct:.0f}%)…",
              flush=True)

        # Use a fresh /tmp working dir — old Joern fails if parsed/ already exists
        work_dir  = Path(f"/tmp/joern_{variant}_{batch_idx}")
        input_dir = work_dir / "tmp"
        out_dir   = work_dir / "parsed" / "tmp"
        shutil.rmtree(work_dir, ignore_errors=True)
        input_dir.mkdir(parents=True)

        for fname in batch:
            shutil.copy2(src_dir / fname, input_dir / fname)

        result = subprocess.run(
            [str(JOERN_PARSE), "tmp"],
            cwd=work_dir,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: joern-parse exited {result.returncode}", flush=True)

        if not out_dir.is_dir():
            print(f"  WARNING: Joern output missing: {out_dir}", flush=True)
            print(result.stderr[:300], flush=True)
            shutil.rmtree(work_dir, ignore_errors=True)
            continue

        # Move per-function subdirs to parsed_cache
        moved = 0
        for sub in out_dir.iterdir():
            if sub.is_dir():
                dest = parsed_cache_v / sub.name
                if not dest.exists():
                    shutil.move(str(sub), str(dest))
                    moved += 1
        print(f"  Moved {moved} parsed dirs to {parsed_cache_v}", flush=True)
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 2: Build GGNN input JSONs
# ---------------------------------------------------------------------------

def build_ggnn_json(mod, model, variant: str, train_names: list, labels: dict):
    out_dir = DEVIGN_INPUT / f"{variant}_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_GGNNinput.json"

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"  {variant} GGNN JSON already exists ({len(existing)} records) — skipping",
              flush=True)
        return len(existing)

    parsed_root = PARSED_CACHE / variant
    rows, skipped = build_graph_json(mod, model, parsed_root, train_names, labels)
    with open(out_path, 'w') as f:
        json.dump(rows, f)
    print(f"  {variant}: {len(rows)} GGNN records saved to {out_path} "
          f"({len(skipped)} skipped)", flush=True)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    split  = json.loads(SPLIT_FILE.read_text())['splits']
    labels = {r['file_name']: int(r['label'])
              for r in json.loads((BASE / 'originals_full_data_with_slices.json').read_text())}

    mod  = load_create_ggnn_module()
    from gensim.models import Word2Vec as _W2V
    wv_path = BASE / "wv_models/devign_full_wv.model"
    model   = _W2V.load(str(wv_path)) if wv_path.exists() else None
    print(f"Word2Vec: {'loaded from ' + str(wv_path) if model else 'NOT FOUND — zero-padded'}", flush=True)
    print(f"Type map: {len(mod.type_map)} node types → {len(mod.type_map)}-dim one-hot", flush=True)

    for variant in VARIANTS:
        print(f"\n{'='*60}", flush=True)
        print(f"Processing {variant}…", flush=True)
        print(f"{'='*60}", flush=True)

        # Step 1: Parse C files
        parse_variant_train(variant, split['train'])

        # Step 2: Build GGNN JSON
        n = build_ggnn_json(mod, model, variant, split['train'], labels)
        print(f"  → {n} records ready", flush=True)

    print("\nREVEAL-Aug data preparation complete!", flush=True)
    print("Next step: run exp7_reveal_aug_train.py", flush=True)


if __name__ == "__main__":
    main()
