#!/usr/bin/env python3
"""
reveal_ren_dead_cf_pipeline.py — Full REVEAL Devign pipeline for pairwise obfuscations.

Computes REVEAL Phase 1+2 results for ren_dead, ren_cf, dead_cf (the three conditions
that require Joern reparsing and Phase 1 retraining).

Pipeline:
  Step 1: Extract test-set functions → individual .c files
  Step 2: Joern-parse them → CPG CSVs
  Step 3: Build GGNN-format JSON from parsed CPGs
  Step 4: Retrain REVEAL Phase 1 GGNN on originals; re-extract embeddings for ALL conditions
  Step 5: Phase 2 metric learning re-eval for ALL 7 conditions; update reveal_7cond_results.json

Usage:
    CUDA_VISIBLE_DEVICES=3 python baselines/reveal/reveal_ren_dead_cf_pipeline.py
"""

import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
THESIS_ROOT = SCRIPT_DIR.parents[1]
BASE        = THESIS_ROOT / "devign_full"
CODE_SLICER = THESIS_ROOT / "data" / "raw" / "ReVeal" / "code-slicer"
CREATE_GGNN = THESIS_ROOT / "data" / "raw" / "ReVeal" / "data_processing" / "create_ggnn_data.py"
REVEAL_RL   = THESIS_ROOT / "data" / "raw" / "ReVeal" / "Vuld_SySe" / "representation_learning"

SPLIT_PATH      = BASE / "devign_full_split_801010.json"
ORIGINALS_TRAIN = BASE / "devign_input" / "originals_train"
AFTER_GGNN      = BASE / "after_ggnn"
RESULTS_PATH    = BASE / "reveal_7cond_results.json"
MODEL_DIR       = SCRIPT_DIR / "models" / "reveal_ggnn_devign_full"
MODEL_CKPT      = MODEL_DIR / "best_model.pt"

# All 3 pairwise variants (slices exist for all three)
PAIRWISE_VARIANTS = ["ren_dead", "ren_cf", "dead_cf"]

# All 7 obfuscation conditions (for Phase 2 re-eval)
ALL_OBF = ["identifier", "deadcode", "controlflow", "compound",
           "ren_dead", "ren_cf", "dead_cf"]

# CUDA_VISIBLE_DEVICES remaps selected GPU to device index 0; always use index 0 when a GPU is visible
CUDA_DEVICE = 0 if torch.cuda.is_available() else -1

# Phase 2 hyperparameters (same as reveal_compound_eval.py)
LAMBDA1    = 0.5
LAMBDA2    = 0.001
ALPHA      = 0.5
HIDDEN_DIM = 256
DROPOUT    = 0.2
NUM_LAYERS = 1
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE   = 5
N_TRIALS   = 10


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(p):
    with open(p) as f:
        return json.load(f)

def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def load_embeddings(json_path):
    data = load_json(json_path)
    X = np.array([d["graph_feature"] for d in data], dtype=np.float32)
    Y = np.array([d["target"]        for d in data], dtype=np.int64)
    return X, Y


# ── Step 1: Extract C files ────────────────────────────────────────────────────
def extract_c_files(variant, test_files, out_dir: Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        existing = sum(1 for _ in out_dir.iterdir())
        print(f"  [skip] {out_dir} has {existing} files")
        return

    slices_path = BASE / f"obf_{variant}_full_data_with_slices.json"
    print(f"  Loading {slices_path} …")
    data = load_json(slices_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in data:
        fname = item.get("file_name", "")
        if fname not in test_files:
            continue
        code = item.get("code", "")
        if not code:
            continue
        (out_dir / fname).write_text(code, encoding="utf-8")
        count += 1

    print(f"  Wrote {count} .c files → {out_dir}")


# ── Step 2: Joern parse ────────────────────────────────────────────────────────
def joern_parse(variant, c_dir: Path, parsed_dir: Path):
    if parsed_dir.exists() and any(parsed_dir.iterdir()):
        n = sum(1 for _ in parsed_dir.iterdir())
        print(f"  [skip] {parsed_dir} has {n} parsed dirs")
        return

    tmp_name = f"tmp_{variant}"
    tmp_dir  = CODE_SLICER / tmp_name
    shutil.rmtree(tmp_dir, ignore_errors=True)

    c_files = list(c_dir.iterdir())
    print(f"  Parsing {len(c_files)} files in batches …")
    parsed_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 2500
    for off in range(0, len(c_files), batch_size):
        batch = c_files[off:off + batch_size]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir()
        for cf in batch:
            shutil.copy2(cf, tmp_dir / cf.name)

        parsed_root = CODE_SLICER / "parsed"
        shutil.rmtree(parsed_root, ignore_errors=True)

        print(f"  Joern batch {off}–{off+len(batch)} …", flush=True)
        t0 = time.time()
        subprocess.run(["./joern/joern-parse", tmp_name],
                       cwd=str(CODE_SLICER), check=True)
        print(f"    Done in {time.time()-t0:.0f}s", flush=True)

        parsed_tmp = parsed_root / tmp_name
        if parsed_tmp.exists():
            for sub in parsed_tmp.iterdir():
                if sub.is_dir():
                    dest = parsed_dir / sub.name
                    if not dest.exists():
                        shutil.move(str(sub), str(dest))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Joern done → {parsed_dir}")


# ── Step 3: Build GGNN JSON ────────────────────────────────────────────────────
def build_ggnn_json(variant, parsed_dir: Path, test_files, labels: dict, ggnn_dir: Path):
    out_path  = ggnn_dir / f"obf_{variant}_test_GGNNinput.json"
    out_graph = ggnn_dir / f"obf_{variant}_test_GGNNinput_graph.json"
    if out_path.exists():
        n = len(load_json(out_path))
        print(f"  [skip] {out_path} ({n} records)")
        return

    spec = importlib.util.spec_from_file_location("create_ggnn_data", str(CREATE_GGNN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wv_path = BASE / "wv_models" / "devign_full_wv.model"
    wv_model = None
    if wv_path.exists():
        from gensim.models import Word2Vec
        wv_model = Word2Vec.load(str(wv_path))
        print(f"  Word2Vec loaded from {wv_path}")
    else:
        print("  WARNING: Word2Vec model not found — node features will be zero-padded")

    rows, skipped = [], []
    for fname in sorted(test_files):
        node_csv = parsed_dir / fname / "nodes.csv"
        edge_csv = parsed_dir / fname / "edges.csv"
        if not node_csv.exists() or not edge_csv.exists():
            skipped.append(fname)
            continue
        gi = mod.inputGeneration(
            str(node_csv), str(edge_csv),
            int(labels.get(fname, 0)), wv_model, mod.edgeType_full, False,
        )
        if gi is None:
            skipped.append(fname)
            continue
        rows.append(gi)

    ggnn_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_path, rows)
    shutil.copy2(out_path, out_graph)
    print(f"  GGNN JSON: {len(rows)} records, {len(skipped)} skipped → {out_path}")


# ── Step 4: Train Phase 1 + re-extract all embeddings ────────────────────────
def train_phase1_and_extract(ggnn_dir: Path):
    device_str = f"cuda:{CUDA_DEVICE}" if CUDA_DEVICE >= 0 and torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    # Import REVEAL GGNN module
    ggnn_script = SCRIPT_DIR / "train_reveal_ggnn.py"
    spec = importlib.util.spec_from_file_location("reveal_ggnn", str(ggnn_script))
    ggnn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ggnn_mod)

    print(f"  Device: {device}")

    # Load originals (already in GGNN format)
    print("  Loading originals …")
    train_graphs, _ = ggnn_mod.load_graphs(ORIGINALS_TRAIN / "train_GGNNinput.json", device)
    valid_graphs, _ = ggnn_mod.load_graphs(ORIGINALS_TRAIN / "valid_GGNNinput.json", device)
    input_dim = train_graphs[0][0].shape[1]
    pos = sum(t for _, _, t in train_graphs)
    neg = len(train_graphs) - pos
    pos_weight = torch.tensor([neg / pos], device=device)
    print(f"  train={len(train_graphs)}  valid={len(valid_graphs)}  dim={input_dim}")

    if MODEL_CKPT.exists():
        print(f"  Loading cached Phase 1 checkpoint …")
        ckpt  = torch.load(str(MODEL_CKPT), map_location=device)
        model = ggnn_mod.REVEALVulnDetector(input_dim=input_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("  Training Phase 1 GGNN …")
        model     = ggnn_mod.REVEALVulnDetector(input_dim=input_dim).to(device)
        loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7, verbose=False)
        model, best_thr, best_f1 = ggnn_mod.train(
            model, train_graphs, valid_graphs, optimizer, scheduler, loss_fn, device,
            max_epochs=100, max_patience=20)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "best_threshold": best_thr,
                    "input_dim": input_dim}, str(MODEL_CKPT))
        print(f"  Saved Phase 1 → {MODEL_CKPT}  (best val F1={best_f1:.2f}%)")

    AFTER_GGNN.mkdir(parents=True, exist_ok=True)

    # Re-extract standard splits
    for split_name, ggnn_json, out_name in [
        ("train",      ORIGINALS_TRAIN / "train_GGNNinput.json", "train_GGNNinput_graph.json"),
        ("valid",      ORIGINALS_TRAIN / "valid_GGNNinput.json", "valid_GGNNinput_graph.json"),
        ("test",       ORIGINALS_TRAIN / "test_GGNNinput.json",  "test_GGNNinput_graph.json"),
    ]:
        out_path = AFTER_GGNN / out_name
        if out_path.exists():
            print(f"  [skip] {out_name}")
            continue
        graphs, _ = ggnn_mod.load_graphs(ggnn_json, device)
        recs = ggnn_mod.extract_embeddings(model, graphs, device)
        save_json(out_path, recs)
        print(f"  Extracted {split_name}: {len(recs)} → {out_name}")

    # Re-extract existing obf conditions
    existing_obf = {
        "obf_identifier":  BASE / "devign_input" / "obf_identifier_test"  / "test_GGNNinput.json",
        "obf_deadcode":    BASE / "devign_input" / "obf_deadcode_test"     / "test_GGNNinput.json",
        "obf_controlflow": BASE / "devign_input" / "obf_controlflow_test"  / "test_GGNNinput.json",
        "obf_compound":    BASE / "devign_input" / "obf_compound_test"     / "test_GGNNinput.json",
    }
    for obf_name, ggnn_path in existing_obf.items():
        out_name = f"{obf_name}_test_GGNNinput_graph.json"
        out_path = AFTER_GGNN / out_name
        if out_path.exists():
            print(f"  [skip] {out_name}")
            continue
        if not ggnn_path.exists():
            print(f"  MISSING: {ggnn_path}")
            continue
        graphs, _ = ggnn_mod.load_graphs(ggnn_path, device)
        recs = ggnn_mod.extract_embeddings(model, graphs, device)
        save_json(out_path, recs)
        print(f"  Extracted {obf_name}: {len(recs)} → {out_name}")

    # Extract pairwise obf conditions
    for variant in PAIRWISE_VARIANTS:
        out_name = f"obf_{variant}_test_GGNNinput_graph.json"
        out_path = AFTER_GGNN / out_name
        if out_path.exists():
            print(f"  [skip] {out_name}")
            continue
        ggnn_path = ggnn_dir / f"obf_{variant}_test_GGNNinput.json"
        if not ggnn_path.exists():
            print(f"  MISSING GGNN input for {variant}: {ggnn_path}")
            continue
        graphs, _ = ggnn_mod.load_graphs(ggnn_path, device)
        recs = ggnn_mod.extract_embeddings(model, graphs, device)
        save_json(out_path, recs)
        print(f"  Extracted {variant}: {len(recs)} → {out_name}")


# ── Step 5: Phase 2 eval ───────────────────────────────────────────────────────
def phase2_eval():
    sys.path.insert(0, str(REVEAL_RL))
    from graph_dataset import DataSet
    from models import MetricLearningModel
    from trainer import train as reveal_train, evaluate as reveal_evaluate

    # Load train pool (train + valid)
    X_train, Y_train = load_embeddings(AFTER_GGNN / "train_GGNNinput_graph.json")
    X_valid, Y_valid = load_embeddings(AFTER_GGNN / "valid_GGNNinput_graph.json")
    X_pool = np.concatenate([X_train, X_valid], axis=0)
    Y_pool = np.concatenate([Y_train, Y_valid], axis=0)
    print(f"  Train pool: {len(X_pool)} (pos={Y_pool.sum()})")

    # Build test_sets dict — canonical name → (X, Y)
    test_sets = {"originals": load_embeddings(AFTER_GGNN / "test_GGNNinput_graph.json")}
    for cond, fname in [
        ("identifier",  "obf_identifier_test_GGNNinput_graph.json"),
        ("deadcode",    "obf_deadcode_test_GGNNinput_graph.json"),
        ("controlflow", "obf_controlflow_test_GGNNinput_graph.json"),
        ("compound",    "obf_compound_test_GGNNinput_graph.json"),
        ("ren_dead",    "obf_ren_dead_test_GGNNinput_graph.json"),
        ("ren_cf",      "obf_ren_cf_test_GGNNinput_graph.json"),
        ("dead_cf",     "obf_dead_cf_test_GGNNinput_graph.json"),
    ]:
        emb_path = AFTER_GGNN / fname
        if emb_path.exists():
            test_sets[cond] = load_embeddings(emb_path)
            print(f"  {cond}: {len(test_sets[cond][0])} samples")
        else:
            print(f"  SKIP {cond}: {emb_path} not found")

    input_dim = X_pool.shape[1]
    all_f1s   = {k: [] for k in test_sets}

    def run_trial(trial_idx):
        seed = 1000 + trial_idx * 7
        np.random.seed(seed); torch.manual_seed(seed)

        X_tr, _, Y_tr, _ = train_test_split(X_pool, Y_pool, test_size=0.2, random_state=seed)

        model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=HIDDEN_DIM, aplha=ALPHA,
            lambda1=LAMBDA1, lambda2=LAMBDA2, dropout_p=DROPOUT, num_layers=NUM_LAYERS,
        )
        optimizer = Adam(model.parameters())
        if CUDA_DEVICE >= 0:
            model.cuda(device=CUDA_DEVICE)

        # Use originals as test during training
        X_orig, Y_orig = test_sets["originals"]
        dataset = DataSet(BATCH_SIZE, input_dim)
        for x, y in zip(X_tr, Y_tr):
            split = 'valid' if np.random.uniform() <= 0.1 else 'train'
            dataset.add_data_entry(x.tolist(), int(y), split)
        for x, y in zip(X_orig, Y_orig):
            dataset.add_data_entry(x.tolist(), int(y), 'test')
        dataset.initialize_dataset(balance=True, output_buffer=None)

        reveal_train(model=model, dataset=dataset, optimizer=optimizer,
                     num_epochs=MAX_EPOCHS, max_patience=PATIENCE,
                     cuda_device=CUDA_DEVICE, output_buffer=None)

        results = {}
        for name, (X_cond, Y_cond) in test_sets.items():
            dataset.clear_test_set()
            for x, y in zip(X_cond, Y_cond):
                dataset.add_data_entry(x.tolist(), int(y), 'test')
            n = dataset.initialize_test_batches()
            acc, pr, rc, f1 = reveal_evaluate(
                model=model, iterator_function=dataset.get_next_test_batch,
                _batch_count=n, cuda_device=CUDA_DEVICE, output_buffer=None)
            results[name] = f1
        return results

    print(f"\n  Running {N_TRIALS} Phase 2 trials …")
    for trial in range(N_TRIALS):
        print(f"  Trial {trial+1}/{N_TRIALS} …", flush=True)
        r = run_trial(trial)
        for k in all_f1s:
            if k in r:
                all_f1s[k].append(r[k])
        print("    " + "  ".join(f"{k}={r.get(k,0):.2f}" for k in list(test_sets.keys())[:4]),
              flush=True)
        print("    " + "  ".join(f"{k}={r.get(k,0):.2f}" for k in list(test_sets.keys())[4:]),
              flush=True)

    # Aggregate
    orig_mean = np.mean(all_f1s["originals"]) if all_f1s["originals"] else 0
    agg = {}
    for k, f1s in all_f1s.items():
        if not f1s:
            continue
        mean = round(float(np.mean(f1s)), 2)
        std  = round(float(np.std(f1s)), 2)
        agg[k] = {"f1": mean, "f1_std": std, "n_trials": len(f1s),
                  "all_f1": [round(v, 2) for v in f1s]}
        if k != "originals":
            agg[k]["delta_f1"] = round(mean - orig_mean, 2)
    return agg


# ── Step 6: Update results JSON ────────────────────────────────────────────────
def update_results(new_agg: dict):
    # Map short names → canonical keys used in reveal_7cond_results.json
    can_map = {
        "originals":   "originals",
        "identifier":  "identifier",
        "deadcode":    "deadcode",
        "controlflow": "controlflow",
        "compound":    "compound",
        "ren_dead":    "ren_dead",
        "ren_cf":      "ren_cf",
        "dead_cf":     "dead_cf",
    }
    existing = load_json(RESULTS_PATH) if RESULTS_PATH.exists() else {}
    for short, canonical in can_map.items():
        if short in new_agg:
            existing[canonical] = new_agg[short]
            v = new_agg[short]
            d = f"  Δ={v.get('delta_f1',0):+.2f}pp" if short != "originals" else ""
            print(f"  {canonical}: F1={v['f1']:.2f}±{v['f1_std']:.2f}%{d}")

    save_json(RESULTS_PATH, existing)
    print(f"\n  → {RESULTS_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("REVEAL Devign Pairwise Pipeline (ren_dead / ren_cf / dead_cf)")
    print("=" * 65)

    print("\n[Setup] Loading split and labels …")
    split      = load_json(SPLIT_PATH)
    test_files = set(split["splits"]["test"])
    orig_pkg   = load_json(BASE / "originals_full_data_with_slices.json")
    labels     = {r["file_name"]: int(r["label"]) for r in orig_pkg}
    print(f"  Test set: {len(test_files)} files")

    ggnn_dir = BASE / "devign_input" / "pairwise_test"
    ggnn_dir.mkdir(parents=True, exist_ok=True)

    # Steps 1–3: build GGNN input for each pairwise variant
    for variant in PAIRWISE_VARIANTS:
        print(f"\n── {variant} (steps 1–3) ──")
        c_dir      = CODE_SLICER / f"c_files_{variant}"
        parsed_dir = BASE / "devign_input" / "parsed_cache" / f"obf_{variant}"

        # Filter test files to those that have sliced code
        slices = load_json(BASE / f"obf_{variant}_full_data_with_slices.json")
        sliced = {item["file_name"] for item in slices if item.get("code")}
        test_v = test_files & sliced
        print(f"  Test files with slices: {len(test_v)}")

        print("  Step 1: Extracting C files …")
        extract_c_files(variant, test_v, c_dir)

        if not (ggnn_dir / f"obf_{variant}_test_GGNNinput.json").exists():
            print("  Step 2: Joern parsing …")
            joern_parse(variant, c_dir, parsed_dir)

            print("  Step 3: Building GGNN JSON …")
            build_ggnn_json(variant, parsed_dir, test_v, labels, ggnn_dir)
        else:
            print(f"  [skip] GGNN JSON already exists")

    # Step 4: Phase 1 training + embedding extraction
    print("\n[Step 4] Phase 1 training + embedding extraction …")
    train_phase1_and_extract(ggnn_dir)

    # Step 5: Phase 2 eval
    print("\n[Step 5] Phase 2 metric learning eval …")
    new_agg = phase2_eval()

    # Step 6: Update results
    print("\n[Step 6] Updating reveal_7cond_results.json …")
    update_results(new_agg)

    print("\n" + "=" * 65)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
