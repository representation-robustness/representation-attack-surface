#!/usr/bin/env python3
"""
Exp 17: Budget-stratified transfer ASR, properly linked to Section 7.2.

For each (source, target) pair, uses the per-function minimum-budget attack
from the greedy attack (Section 7.2) and measures whether the same attack
also evades the target detector.

Implements the professor's TASR formula extended to budget levels:

  TASR_k(f_s -> f_t) =
      |{x in X_{s,t}^succ : f_t(A_k(x; f_s)) = 0}|
      -----------------------------------------------
               |X_{s,t}^succ|

where X_{s,t}   = {x : y(x)=1, f_s(x)=1, f_t(x)=1}  (both correct)
      X_{s,t}^succ = {x in X_{s,t} : f_s(A(x;f_s))=0} (source-evadeable)
and A_k(x; f_s) is the source's minimum-budget attack truncated to budget<=k.

Section 7.2 is the diagonal case (f_s = f_t), uses ASR@k from Sec 5.1.
Section 7.3 is the off-diagonal case, uses TASR from Sec 5.2.

Alignment:
- Graph-family (ANGLE, VulGNN, ECG-RGCN): CPG test set, 5311 functions
- Seq-family (CodeBERT, CodeT5+, ReGVD): text test set, 2687 functions
- REVEAL: GGNN test set, 2670 functions — a verified subsequence of the text
  test set (17 functions dropped during GGNN graph construction). Aligned to
  the text test set via reveal_text_index_map.json. REVEAL has no B2 pairwise
  conditions (ren_dead, ren_cf, dead_cf); min-budget attacks are B1 or B3.

Output: devign_full/transfer_v2_results.json
"""

import json
import numpy as np
from pathlib import Path

THESIS        = Path(__file__).resolve().parents[1]
PREDS_DIR     = THESIS / "devign_full" / "attack" / "preds"
MAP_FILE      = THESIS / "devign_full" / "attack" / "cpg_text_index_map.json"
REV_MAP_FILE  = THESIS / "devign_full" / "attack" / "reveal_text_index_map.json"
OUT_FILE      = THESIS / "devign_full" / "transfer_v2_results.json"

# Budget levels (priority order within each budget)
B1_CONDS = ["ren", "dead", "cf"]
B2_CONDS = ["ren_dead", "ren_cf", "dead_cf"]
B3_CONDS = ["compound"]

# Renaming-invariant models use pairwise proxy conditions
RENAMING_INVARIANT = {"ecg_rgcn", "vulgnn"}
INVARIANCE_MAP = {"ren_dead": "dead", "ren_cf": "cf", "dead_cf": "compound"}

# Family membership for alignment decisions
GRAPH_FAMILY  = {"angle", "vulgnn", "ecg_rgcn"}   # CPG test set, 5311 functions
SEQ_FAMILY    = {"codebert", "codet5plus", "regvd"} # text test set, 2687 functions
REVEAL_FAMILY = {"reveal"}                          # GGNN test set, 2670 functions

SOURCE_MODELS = ["angle", "vulgnn", "ecg_rgcn", "reveal", "codebert", "codet5plus", "regvd"]
TARGET_MODELS = ["angle", "vulgnn", "ecg_rgcn", "reveal", "codebert", "codet5plus", "regvd"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_preds(name):
    path = PREDS_DIR / f"{name}_preds.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_cond_preds(seed_preds, cond, n, model_name):
    """Return predictions for cond, applying invariance map for ren-invariant models."""
    if cond in seed_preds:
        return seed_preds[cond][:n]
    if model_name in RENAMING_INVARIANT and cond in INVARIANCE_MAP:
        proxy = INVARIANCE_MAP[cond]
        if proxy in seed_preds:
            return seed_preds[proxy][:n]
    return [1] * n  # not available → assume not evaded


def min_budget_attacks_for_seed(seed_preds, true_labels, indices, model_name):
    """
    For each function index in `indices`, find the minimum-budget (B1→B2→B3)
    condition that evades the model under this seed.

    Returns dict: idx -> (budget, cond)  — only for functions that CAN be evaded.
    Correctly-classified-on-clean filter is applied using `indices`.
    """
    n = len(true_labels)
    clean = seed_preds.get("clean", [])[:n]

    # Within `indices`, keep only those correctly classified as vulnerable
    vuln_correct = [i for i in indices if i < len(clean) and
                    true_labels[i] == 1 and clean[i] == 1]

    attacks = {}
    for i in vuln_correct:
        found = False
        for cond in B1_CONDS:
            p = get_cond_preds(seed_preds, cond, n, model_name)
            if i < len(p) and p[i] == 0:
                attacks[i] = (1, cond)
                found = True
                break
        if found:
            continue
        # model has no B2 conditions (e.g. REVEAL) — check if available
        has_b2 = any(c in seed_preds or
                     (model_name in RENAMING_INVARIANT and INVARIANCE_MAP.get(c) in seed_preds)
                     for c in B2_CONDS)
        if has_b2:
            for cond in B2_CONDS:
                p = get_cond_preds(seed_preds, cond, n, model_name)
                if i < len(p) and p[i] == 0:
                    attacks[i] = (2, cond)
                    found = True
                    break
        if found:
            continue
        for cond in B3_CONDS:
            p = get_cond_preds(seed_preds, cond, n, model_name)
            if i < len(p) and p[i] == 0:
                attacks[i] = (3, cond)
                break

    return vuln_correct, attacks


def compute_transfer(src_data, tgt_data, src_name, tgt_name,
                     src_indices, tgt_indices, true_labels_src, true_labels_tgt,
                     idx_src_to_tgt=None, idx_tgt_to_src=None):
    """
    Compute budget-stratified ASR_k(src -> tgt).

    src_indices: which indices (in src space) to consider as the pool
    tgt_indices: which indices (in tgt space) to consider as the pool
    idx_src_to_tgt: optional mapping from src index to tgt index (cross-family)
    idx_tgt_to_src: optional mapping from tgt index to src index (cross-family)

    Returns dict with asr1, asr2, asr3, mean_transfer_budget, n_pool.
    """
    s_seeds = src_data.get("seeds", {})
    t_seeds = tgt_data.get("seeds", {})

    all_asr1, all_asr2, all_asr3, all_budgets = [], [], [], []

    for s_key, s_seed_preds in s_seeds.items():
        n_src = len(true_labels_src)
        vuln_correct_s, attacks_s = min_budget_attacks_for_seed(
            s_seed_preds, true_labels_src, src_indices, src_name
        )

        for t_key, t_seed_preds in t_seeds.items():
            n_tgt = len(true_labels_tgt)
            t_clean = t_seed_preds.get("clean", [])[:n_tgt]

            # Build X_{s,t}: both correctly classify on clean
            both_correct = []
            for s_idx in vuln_correct_s:
                # Map src index to tgt index
                if idx_src_to_tgt is not None:
                    t_idx = idx_src_to_tgt.get(s_idx)
                    if t_idx is None:
                        continue
                else:
                    t_idx = s_idx  # same-family: indices are identical

                if t_idx >= len(t_clean):
                    continue
                if true_labels_tgt[t_idx] == 1 and t_clean[t_idx] == 1:
                    both_correct.append((s_idx, t_idx))

            if not both_correct:
                continue

            # For each function in X_{s,t}, apply source's min-budget attack to target
            evaded1, evaded2, evaded3 = [], [], []
            budgets_transferred = []

            for (s_idx, t_idx) in both_correct:
                if s_idx not in attacks_s:
                    continue  # source couldn't evade this function at any budget

                budget, cond = attacks_s[s_idx]
                t_preds = get_cond_preds(t_seed_preds, cond, n_tgt, tgt_name)
                target_evaded = (t_idx < len(t_preds) and t_preds[t_idx] == 0)

                if target_evaded:
                    if budget == 1:
                        evaded1.append(s_idx)
                        evaded2.append(s_idx)
                        evaded3.append(s_idx)
                    elif budget == 2:
                        evaded2.append(s_idx)
                        evaded3.append(s_idx)
                    elif budget == 3:
                        evaded3.append(s_idx)
                    budgets_transferred.append(budget)

            # Professor's TASR: denominator = |X_{s,t}^succ| (source-evadeable only)
            n_succ = sum(1 for (s_idx, _) in both_correct if s_idx in attacks_s)
            if n_succ == 0:
                continue
            all_asr1.append(len(evaded1) / n_succ * 100)
            all_asr2.append(len(evaded2) / n_succ * 100)
            all_asr3.append(len(evaded3) / n_succ * 100)
            if budgets_transferred:
                all_budgets.extend(budgets_transferred)

    if not all_asr1:
        return None

    return {
        "asr1": round(float(np.mean(all_asr1)), 2),
        "asr2": round(float(np.mean(all_asr2)), 2),
        "asr3": round(float(np.mean(all_asr3)), 2),
        "mean_transfer_budget": round(float(np.mean(all_budgets)), 2) if all_budgets else None,
        "n_seed_pairs": len(all_asr1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def get_family(name):
    if name in GRAPH_FAMILY:  return "graph"
    if name in REVEAL_FAMILY: return "reveal"
    return "seq"


def resolve_indices(src_name, tgt_name,
                    cpg_to_text, text_to_cpg,
                    text_to_reveal, reveal_to_text,
                    src_labels):
    """
    Return (src_indices, idx_src_to_tgt) for the (src, tgt) pair.

    Index spaces:
      graph  → CPG space  (0..5310)
      seq    → text space (0..2686)
      reveal → reveal space (0..2669)

    Cross-space mappings available:
      cpg ↔ text   (cpg_to_text / text_to_cpg)
      text ↔ reveal (text_to_reveal / reveal_to_text)
      cpg ↔ reveal  (chain: cpg→text→reveal / reveal→text→cpg)
    """
    sf = get_family(src_name)
    tf = get_family(tgt_name)

    if sf == tf:
        return list(range(len(src_labels))), None

    # graph ↔ seq
    if sf == "graph" and tf == "seq":
        return list(cpg_to_text.keys()), cpg_to_text
    if sf == "seq" and tf == "graph":
        return list(text_to_cpg.keys()), text_to_cpg

    # graph ↔ reveal  (chain through text space)
    if sf == "graph" and tf == "reveal":
        # src idx in CPG space → text idx → reveal idx
        cpg_to_reveal = {ci: text_to_reveal[ti]
                         for ci, ti in cpg_to_text.items()
                         if ti in text_to_reveal}
        return list(cpg_to_reveal.keys()), cpg_to_reveal
    if sf == "reveal" and tf == "graph":
        # src idx in reveal space → text idx → CPG idx
        reveal_to_cpg = {ri: text_to_cpg[ti]
                         for ri, ti in reveal_to_text.items()
                         if ti in text_to_cpg}
        return list(reveal_to_cpg.keys()), reveal_to_cpg

    # seq ↔ reveal
    if sf == "seq" and tf == "reveal":
        # text idx → reveal idx (only for the 2670 that survived GGNN)
        return list(text_to_reveal.keys()), text_to_reveal
    if sf == "reveal" and tf == "seq":
        # reveal idx → text idx
        return list(reveal_to_text.keys()), reveal_to_text

    raise ValueError(f"Unhandled family pair: {sf} → {tf}")


def main():
    # Load cross-family index mappings
    mapping  = json.loads(MAP_FILE.read_text())
    cpg_to_text  = {int(k): v for k, v in mapping["cpg_to_text"].items()}
    text_to_cpg  = {int(k): v for k, v in mapping["text_to_cpg"].items()}

    rev_mapping  = json.loads(REV_MAP_FILE.read_text())
    text_to_reveal = {int(k): v for k, v in rev_mapping["text_to_reveal"].items()}
    reveal_to_text = {int(k): v for k, v in rev_mapping["reveal_to_text"].items()}

    # Load all prediction files
    all_preds = {}
    all_true_labels = {}
    for name in SOURCE_MODELS + [m for m in TARGET_MODELS if m not in SOURCE_MODELS]:
        d = load_preds(name)
        if d is None:
            print(f"  MISSING: {name}_preds.json", flush=True)
            continue
        all_preds[name] = d
        all_true_labels[name] = d["true_labels"]
        n_seeds = len(d.get("seeds", {}))
        conds = list(list(d["seeds"].values())[0].keys()) if d.get("seeds") else []
        print(f"Loaded {name}: {len(d['true_labels'])} functions, {n_seeds} seeds, "
              f"conds={conds}", flush=True)

    results = {}

    for src_name in SOURCE_MODELS:
        if src_name not in all_preds:
            continue
        src_data   = all_preds[src_name]
        src_labels = all_true_labels[src_name]
        src_family = get_family(src_name)

        results[src_name] = {}

        for tgt_name in TARGET_MODELS:
            if tgt_name not in all_preds or tgt_name == src_name:
                results[src_name][tgt_name] = None
                continue

            tgt_data   = all_preds[tgt_name]
            tgt_labels = all_true_labels[tgt_name]
            tgt_family = get_family(tgt_name)

            print(f"\n{src_name} -> {tgt_name}  [{src_family}->{tgt_family}]", flush=True)

            src_indices, idx_src_to_tgt = resolve_indices(
                src_name, tgt_name,
                cpg_to_text, text_to_cpg,
                text_to_reveal, reveal_to_text,
                src_labels,
            )

            r = compute_transfer(
                src_data, tgt_data, src_name, tgt_name,
                src_indices, None, src_labels, tgt_labels,
                idx_src_to_tgt=idx_src_to_tgt,
                idx_tgt_to_src=None,
            )
            results[src_name][tgt_name] = r

            if r:
                print(f"  ASR@1={r['asr1']:.2f}%  ASR@2={r['asr2']:.2f}%  "
                      f"ASR@3={r['asr3']:.2f}%  mean_budget={r['mean_transfer_budget']}  "
                      f"seed_pairs={r['n_seed_pairs']}", flush=True)
            else:
                print("  No result (no valid seed pairs)", flush=True)

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT_FILE}", flush=True)

    # ── print summary table ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TRANSFER ASR@1 MATRIX (budget-stratified, linked to Sec 7.2)")
    print("="*80)
    hdr = f"{'Source':<18}" + "".join(f"{t:>12}" for t in TARGET_MODELS)
    print(hdr)
    print("-"*80)
    for src in SOURCE_MODELS:
        row = f"{src:<18}"
        for tgt in TARGET_MODELS:
            if tgt == src:
                row += f"{'--':>12}"
            elif results.get(src, {}).get(tgt) is None:
                row += f"{'N/A':>12}"
            else:
                row += f"{results[src][tgt]['asr1']:>11.2f}%"
        print(row)

    print("\nASR@3 MATRIX (cumulative max transfer)")
    print("-"*80)
    for src in SOURCE_MODELS:
        row = f"{src:<18}"
        for tgt in TARGET_MODELS:
            if tgt == src:
                row += f"{'--':>12}"
            elif results.get(src, {}).get(tgt) is None:
                row += f"{'N/A':>12}"
            else:
                row += f"{results[src][tgt]['asr3']:>11.2f}%"
        print(row)


if __name__ == "__main__":
    main()
