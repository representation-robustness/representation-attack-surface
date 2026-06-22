#!/usr/bin/env python3
"""
greedy_rename_attack.py

Addresses professor comments lines 342-347 of conference-paper.tex:
  "need optimization-based attack (even heuristic); adaptive attack targeting
   specific detector; and budgeted perturbation (minimize edits)"

Greedy budgeted identifier renaming attack against fine-tuned CodeBERT.

For each test function, every identifier is scored by how much renaming it
(alone) reduces the vulnerability logit.  Top-k identifiers by score are
renamed — a heuristic greedy approximation to the min-edit evasion problem.

Compared against: random k-rename (5 trials) and blanket rename (all ids).

Trains CodeBERT seed 42 and saves checkpoint if not present (~30-60 min).

Output: ~/thesis/devign_full/greedy_attack_results.json
"""
import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaForSequenceClassification, RobertaTokenizer

DEVIGN_ROOT  = Path.home() / "thesis/devign_full"
SPLIT_FILE   = DEVIGN_ROOT / "devign_full_split_801010.json"
CKPT_PATH    = Path.home() / "thesis/baselines/codebert/ckpts_multiseed/codebert_seed42.pt"
DATA_FILES   = {
    "originals":      DEVIGN_ROOT / "originals_full_data_with_slices.json",
    "obf_identifier": DEVIGN_ROOT / "obf_identifier_full_data_with_slices.json",
}
MODEL_NAME   = "microsoft/codebert-base"
DEVICE       = torch.device("cuda:0")
MAX_LENGTH   = 512
BATCH_SIZE   = 16
NUM_EPOCHS   = 10
PATIENCE     = 3
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BUDGETS      = [1, 2, 3, 5]
RANDOM_TRIALS = 5

C_KEYWORDS = {
    "auto","break","case","char","const","continue","default","do","double","else",
    "enum","extern","float","for","goto","if","inline","int","long","register",
    "restrict","return","short","signed","sizeof","static","struct","switch",
    "typedef","union","unsigned","void","volatile","while","NULL","true","false",
    "size_t","uint8_t","uint16_t","uint32_t","uint64_t","int8_t","int16_t",
    "int32_t","int64_t","uint","ulong","ushort","uchar","bool","ptrdiff_t",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_identifiers(code):
    """Unique non-keyword identifier tokens, in order of first appearance."""
    seen, seen_set = [], set()
    for t in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code):
        if t not in C_KEYWORDS and not t.startswith("__") and t not in seen_set:
            seen.append(t); seen_set.add(t)
    return seen


def rename_identifiers(code, ids_to_rename):
    result = code
    for i, ident in enumerate(ids_to_rename):
        result = re.sub(r"\b" + re.escape(ident) + r"\b",
                        f"__v_{i+1:04d}", result)
    return result


# ── Dataset / DataLoader ──────────────────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(r["code"], max_length=MAX_LENGTH,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(int(r["label"]), dtype=torch.long)}


def make_loader(records, tokenizer, bs, balanced=False):
    ds = CodeDataset(records, tokenizer)
    if balanced:
        labels = [int(r["label"]) for r in records]
        pos = sum(labels); neg = len(labels) - pos
        w = [1/neg if l == 0 else 1/pos for l in labels]
        return DataLoader(ds, batch_size=bs,
                          sampler=WeightedRandomSampler(w, len(w), replacement=True),
                          num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=4, pin_memory=True)


# ── Training ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_f1(model, loader):
    model.eval()
    preds, truths = [], []
    for b in loader:
        logits = model(input_ids=b["input_ids"].to(DEVICE),
                       attention_mask=b["attention_mask"].to(DEVICE)).logits
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(b["labels"].tolist())
    return f1_score(truths, preds, zero_division=0) * 100, preds, truths


def train_and_save(tokenizer, train_recs, valid_recs):
    print("Training CodeBERT seed 42 (checkpoint will be saved for re-use)...")
    set_seed(42)
    train_loader = make_loader(train_recs, tokenizer, BATCH_SIZE, balanced=True)
    valid_loader = make_loader(valid_recs, tokenizer, BATCH_SIZE * 2)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)
    total_steps  = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps),
    ], milestones=[warmup_steps])

    best_val_f1, best_state, no_improve = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train(); total_loss = n = 0
        for b in train_loader:
            loss = model(input_ids=b["input_ids"].to(DEVICE),
                         attention_mask=b["attention_mask"].to(DEVICE),
                         labels=b["labels"].to(DEVICE)).loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item(); n += 1
        val_f1, _, _ = eval_f1(model, valid_loader)
        print(f"  Ep {epoch} loss={total_loss/n:.4f} val_F1={val_f1:.2f}%", flush=True)
        if val_f1 > best_val_f1 + 0.1:
            best_val_f1, best_state, no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CKPT_PATH)
    print(f"  Checkpoint saved → {CKPT_PATH}  (best val_F1={best_val_f1:.2f}%)")
    return model


# ── Attack ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_vuln_logit(model, tokenizer, codes, bs=32):
    """Return vulnerability logit (class-1 logit) for each code string."""
    logits = []
    for i in range(0, len(codes), bs):
        enc = tokenizer(codes[i:i+bs], max_length=MAX_LENGTH,
                        padding=True, truncation=True, return_tensors="pt")
        out = model(input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE)).logits
        logits.extend(out[:, 1].cpu().tolist())
    return logits


def score_all_identifiers(model, tokenizer, test_recs):
    """
    Score every identifier in every test function.
    Returns (all_scores, orig_logits) where all_scores[i] = {id: impact_score}.
    impact_score > 0 means renaming this id reduces the vulnerability logit.
    """
    orig_codes = [r["code"] for r in test_recs]
    print(f"  Computing baseline vulnerability logits ({len(orig_codes)} functions)...")
    orig_logits = batch_vuln_logit(model, tokenizer, orig_codes)

    # Build flat list of all (fn_idx, identifier, variant_code) triples
    fn_ids    = []   # which function each variant belongs to
    id_names  = []   # which identifier
    variants  = []   # the code variant
    for fn_idx, r in enumerate(test_recs):
        for ident in extract_identifiers(r["code"]):
            variant = re.sub(r"\b" + re.escape(ident) + r"\b", "__v_0001", r["code"])
            fn_ids.append(fn_idx)
            id_names.append(ident)
            variants.append(variant)

    print(f"  Scoring {len(variants)} identifier variants across {len(test_recs)} functions...")
    var_logits = batch_vuln_logit(model, tokenizer, variants, bs=32)

    all_scores = [{} for _ in test_recs]
    for var_idx, (fn_idx, ident) in enumerate(zip(fn_ids, id_names)):
        all_scores[fn_idx][ident] = orig_logits[fn_idx] - var_logits[var_idx]

    return all_scores, orig_logits


def evaluate_attacked(model, tokenizer, test_recs, all_scores,
                       orig_logits, baseline_preds, labels, strategy, k):
    """Apply attack with given strategy and budget k, return F1 and ASR."""
    attacked_codes = []
    for r, scores in zip(test_recs, all_scores):
        idents = list(scores.keys())
        if not idents:
            attacked_codes.append(r["code"]); continue

        if strategy == "greedy":
            ranked = sorted(idents, key=lambda x: scores[x], reverse=True)
            to_rename = ranked if k == "all" else ranked[:k]
        elif strategy == "random":
            shuffled = idents.copy(); random.shuffle(shuffled)
            to_rename = shuffled[:k]
        else:
            to_rename = idents  # blanket

        attacked_codes.append(rename_identifiers(r["code"], to_rename))

    attacked_recs = [{"code": c, "label": r["label"]}
                     for c, r in zip(attacked_codes, test_recs)]
    loader = make_loader(attacked_recs, tokenizer, BATCH_SIZE * 2)
    f1, preds, _ = eval_f1(model, loader)

    tp_idx = [i for i, (p, l) in enumerate(zip(baseline_preds, labels))
              if p == 1 and l == 1]
    asr = 100 * sum(1 for i in tp_idx if preds[i] == 0) / len(tp_idx) if tp_idx else 0.0
    return round(float(f1), 2), round(asr, 2)


def run_attack(model, tokenizer, test_recs):
    labels = [int(r["label"]) for r in test_recs]
    all_scores, orig_logits = score_all_identifiers(model, tokenizer, test_recs)

    baseline_preds = [1 if v > 0 else 0 for v in orig_logits]
    baseline_f1 = round(f1_score(labels, baseline_preds, zero_division=0) * 100, 2)
    print(f"\n  Baseline F1: {baseline_f1:.2f}%")

    results = {"baseline_f1": baseline_f1, "greedy": {}, "random": {}}

    # Greedy attack
    print("\n  Greedy attack:")
    for k in BUDGETS + ["all"]:
        f1, asr = evaluate_attacked(model, tokenizer, test_recs, all_scores,
                                     orig_logits, baseline_preds, labels, "greedy", k)
        results["greedy"][f"k={k}"] = {
            "f1": f1, "delta_f1": round(f1 - baseline_f1, 2), "asr": asr}
        print(f"    k={k:<4}  F1={f1:.2f}%  Δ={f1-baseline_f1:+.2f}pp  ASR={asr:.1f}%",
              flush=True)

    # Random rename baseline
    print("\n  Random rename baseline:")
    for k in BUDGETS:
        f1s, asrs = [], []
        for trial in range(RANDOM_TRIALS):
            random.seed(trial * 13 + k)
            f1, asr = evaluate_attacked(model, tokenizer, test_recs, all_scores,
                                         orig_logits, baseline_preds, labels, "random", k)
            f1s.append(f1); asrs.append(asr)
        results["random"][f"k={k}"] = {
            "f1_mean": round(np.mean(f1s), 2), "f1_std": round(np.std(f1s), 2),
            "delta_f1": round(np.mean(f1s) - baseline_f1, 2),
            "asr_mean": round(np.mean(asrs), 2),
        }
        print(f"    k={k}  F1={np.mean(f1s):.2f}±{np.std(f1s):.2f}%  "
              f"Δ={np.mean(f1s)-baseline_f1:+.2f}pp  ASR={np.mean(asrs):.1f}%",
              flush=True)

    # Load blanket rename delta from existing results for comparison
    try:
        existing = json.load(open(DEVIGN_ROOT / "codebert_multiseed_results.json"))
        results["blanket_rename"] = {
            "f1":      existing["test_obf_identifier"]["f1_mean"],
            "delta_f1": existing["test_obf_identifier"]["delta_f1"],
        }
    except Exception:
        pass

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Greedy Budgeted Identifier Renaming Attack")
    print("=" * 60 + "\n")

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    with open(DATA_FILES["originals"]) as f:
        orig = json.load(f)
    file_idx = {d["file_name"]: d for d in orig}

    train_recs = [file_idx[n] for n in split["splits"]["train"] if n in file_idx]
    valid_recs = [file_idx[n] for n in split["splits"]["valid"] if n in file_idx]
    test_recs  = [file_idx[n] for n in split["splits"]["test"]  if n in file_idx]
    print(f"train={len(train_recs)}  valid={len(valid_recs)}  test={len(test_recs)}\n")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    if CKPT_PATH.exists():
        print(f"Loading checkpoint {CKPT_PATH}...")
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2).to(DEVICE)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    else:
        model = train_and_save(tokenizer, train_recs, valid_recs)

    results = run_attack(model, tokenizer, test_recs)

    out_path = DEVIGN_ROOT / "greedy_attack_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print(f"\n{'Budget':<8} {'Greedy F1':>10} {'Greedy Δ':>10} {'ASR':>7}"
          f"  {'Random F1':>10} {'Random Δ':>10} {'ASR':>7}")
    print("-" * 68)
    for k in BUDGETS:
        ks = f"k={k}"
        g = results["greedy"][ks]
        r = results["random"][ks]
        print(f"  {ks:<6} {g['f1']:>10.2f} {g['delta_f1']:>+10.2f} {g['asr']:>6.1f}%"
              f"  {r['f1_mean']:>10.2f} {r['delta_f1']:>+10.2f} {r['asr_mean']:>6.1f}%")
    g_all = results["greedy"].get("k=all", {})
    bl = results.get("blanket_rename", {})
    print(f"  {'k=all':<6} {g_all.get('f1',0):>10.2f} {g_all.get('delta_f1',0):>+10.2f} "
          f"{g_all.get('asr',0):>6.1f}%  "
          f"(blanket={bl.get('f1','?')} Δ={bl.get('delta_f1','?')}pp)")


if __name__ == "__main__":
    main()
