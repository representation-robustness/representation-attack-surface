#!/usr/bin/env python3
"""Aggregate LMGGNN per-seed 7-cond results into multiseed summary."""
import json, numpy as np
from pathlib import Path

RESULT_DIR = Path(__file__).parent
SEEDS = [42, 1337, 7, 100, 999]
CONDS = ['original','identifier','deadcode','controlflow','ren_dead','ren_cf','dead_cf','compound']

for dataset in ['bigvul', 'diversevul']:
    avail = []
    for s in SEEDS:
        p = RESULT_DIR / f'{dataset}_lmggnn_seed{s}_results.json'
        if p.exists():
            d = json.load(open(p))
            if 'compound' in d:  # only use 7-cond results
                avail.append((s, d))
    if not avail:
        print(f'{dataset}: no 7-cond seeds yet')
        continue
    seeds_used = [s for s,_ in avail]
    print(f'\n{dataset}  seeds={seeds_used}')
    agg = {'n_seeds': len(avail), 'seeds': seeds_used, 'model': f'LMGGNN-{dataset}'}
    base_f1s = [d['original']['f1'] for _,d in avail]
    base = np.mean(base_f1s)
    for c in CONDS:
        f1s = [d[c]['f1'] for _,d in avail if c in d]
        if not f1s:
            continue
        agg[c] = {
            'f1_mean':  round(float(np.mean(f1s)), 2),
            'f1_std':   round(float(np.std(f1s)),  2),
            'acc_mean': round(float(np.mean([d[c]['acc'] for _,d in avail if c in d])), 2),
            'all_f1': f1s,
        }
        if c != 'original':
            agg[c]['delta_f1'] = round(agg[c]['f1_mean'] - agg['original']['f1_mean'], 2)
        delta_str = f"  Δ={agg[c].get('delta_f1',0):+.2f}" if c != 'original' else ''
        print(f'  {c:<15} F1={agg[c]["f1_mean"]:.2f} ± {agg[c]["f1_std"]:.2f}{delta_str}')
    out = RESULT_DIR / f'{dataset}_lmggnn_multiseed_results.json'
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'  -> {out}')
