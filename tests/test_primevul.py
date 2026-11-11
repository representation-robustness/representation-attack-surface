"""Dataset isolation and partial-output regressions for the PrimeVul extension."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import plan
from pipeline.data import prepare
from pipeline.report import tables


class TestPrimeVul(unittest.TestCase):
    def test_loader_preserves_order_labels_and_other_datasets(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prepare, 'ROOT', Path(tmp)):
            root = Path(tmp)
            raw = root / 'data/primevul/raw'
            raw.mkdir(parents=True)
            frozen = root / 'data/devign/clean/train.jsonl'
            frozen.parent.mkdir(parents=True)
            frozen.write_bytes(b'frozen\n')
            for split in prepare.SPLITS:
                rows = [{'idx': 20, 'func': 'int f(){return 0;}', 'target': 0},
                        {'idx': 3, 'func': 'int g(){return 1;}', 'target': 1}]
                (raw / f'primevul_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            prepare.primevul()
            for split in prepare.SPLITS:
                out = root / f'data/primevul/clean/{split}.jsonl'
                rows = [json.loads(l) for l in out.read_text().splitlines()]
                self.assertEqual([r['id'] for r in rows], [f'primevul_{split}_000000', f'primevul_{split}_000001'])
                self.assertEqual([r['target'] for r in rows], [0, 1])
                man = json.loads((out.parent / 'manifest.json').read_text())
                self.assertEqual(man['splits'][split]['sha256'], prepare.sha256(out))
            self.assertEqual(frozen.read_bytes(), b'frozen\n')

    def test_primevul_cpu_and_gpu_isolation(self):
        with patch.dict(os.environ, {'RERUN_DATASETS': 'primevul', 'RERUN_SEEDS': '42', 'RERUN_JOBS': ''}):
            cpu = plan.cpu_jobs()
            names = {name for _, name, _, _ in cpu}
            self.assertIn('prepare_primevul', names)
            self.assertNotIn('prepare_clean', names)
            self.assertFalse(any('devign' in n or 'bigvul' in n or 'diversevul' in n for n in names))
            jobs = [j for j in plan.gpu_jobs() if plan.selected(j[0])]
            self.assertTrue(jobs)
            for name, cmd, deps in jobs:
                self.assertTrue(name.endswith('_primevul_s42'))
                self.assertNotIn('--train-conditions', cmd)
                self.assertTrue(set(deps) <= names)
            text = next(j for j in jobs if j[0] == 'codebert_primevul_s42')
            self.assertFalse(any(d.startswith('graphs_') for d in text[2]))

    def test_missing_validity_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(tables, 'ROOT', root), patch.object(tables, 'OUT', root):
                tables.validity_table()
                self.assertEqual((root / 'table_validity.csv').read_text().count('\n'), 1)

    def test_partial_train_with_one_class(self):
        from pipeline.models import common
        from pipeline.report import analyses
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'data/primevul/clean/train.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'id': 'x', 'func': 'int f(){return 0;}', 'target': 0}) + '\n')
            with patch.object(analyses, 'ROOT', root), patch.object(common, 'ROOT', root), patch.object(analyses, 'OUT', root):
                analyses.pattern_density()
            self.assertIn('n_functions,0,1', (root / 'table_pattern_density.csv').read_text())

    def test_cf_sensitivity_single_seed_and_existing_significance(self):
        import gzip
        from pipeline.eval import metrics
        from pipeline.report import analyses
        conds = ['controlflow', 'ren_cf', 'dead_cf', 'compound']
        ids = ['a', 'b', 'c', 'd']
        clean = {'ids': ids, 'y': [0, 0, 1, 1], 'p': [0.1, 0.2, 0.8, 0.9]}
        changed = [0.1, 0.2, 0.1, 0.9]
        cases = {'codebert': [changed], 'regvd': [changed] * 3,
                 'angle': [changed, [0.9, 0.2, 0.1, 0.9], [0.9, 0.9, 0.1, 0.9]],
                 'tfidf_lr': [changed] * 3}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, probs in cases.items():
                d = root / 'results/preds/primevul' / model
                d.mkdir(parents=True)
                for i, prob in enumerate(probs):
                    test = {'clean': clean, **{c: dict(clean, p=prob) for c in conds}}
                    with gzip.open(d / f'run_{i}.json.gz', 'wt') as f:
                        json.dump({'threshold': 0.5, 'test': test}, f)
            st = {(model, c): {'delta_f1': -33.333, 'significant': False}
                  for model in cases for c in conds}
            with patch.object(analyses, 'ROOT', root), patch.object(analyses, 'OUT', root), \
                    patch.object(analyses, 'MAIN', list(cases)), \
                    patch.object(analyses, 'stat_rows', return_value=st), \
                    patch.object(analyses, 'cf_fallback_ids', return_value=set()), \
                    patch.object(metrics, 'valid_ids', return_value=set(ids)):
                analyses.cf_sensitivity_table('primevul')
            tex = (root / 'table_cf_sensitivity_primevul.tex').read_text().splitlines()
            for model in cases:
                row = next(line for line in tex if line.startswith(analyses.NAMES[model] + ' &'))
                self.assertEqual('\\mathbf' in row, model in ('angle', 'tfidf_lr'), model)
            csv = (root / 'table_cf_sensitivity_primevul.csv').read_text()
            self.assertIn('primevul,codebert,compound,4,-33.3333,,', csv)
