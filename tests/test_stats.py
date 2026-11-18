"""Statistics used by every table: BH adjustment, paired t-test, BH families, calibration."""
import unittest

import numpy as np
from scipy import stats as sps

from pipeline.eval.metrics import ece, scores
from pipeline.eval.stats import bh, bh_family


class TestBH(unittest.TestCase):
    def test_known_values(self):
        # Benjamini & Hochberg (1995) step-up: p_(i) * m / i, made monotone from the largest p down.
        p = [0.01, 0.04, 0.03, 0.005]
        self.assertTrue(np.allclose(bh(p), [0.02, 0.04, 0.04, 0.02]))

    def test_monotone_and_bounded(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200)
        q = np.array(bh(p))
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))
        self.assertTrue(np.all((q >= p - 1e-12) & (q <= 1)))

    def test_matches_scipy(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(size=50) ** 3
        self.assertTrue(np.allclose(bh(p), sps.false_discovery_control(p, method="bh")))


class TestFamilies(unittest.TestCase):
    def test_families(self):
        self.assertEqual(bh_family("reveal"), "main")
        self.assertEqual(bh_family("reveal_struct"), "ablation")
        self.assertEqual(bh_family("vulgnn_struct_cb"), "ablation")
        self.assertEqual(bh_family("codebert_aug"), "training")
        self.assertEqual(bh_family("regvd_aug_ren+dead"), "training")
        self.assertEqual(bh_family("reveal_volctl"), "training")


class TestPairedT(unittest.TestCase):
    def test_one_sample_on_deltas_is_paired_t(self):
        a, b = np.array([60.1, 61.0, 59.5, 60.7, 60.2]), np.array([58.0, 59.9, 58.1, 59.0, 58.8])
        self.assertAlmostEqual(sps.ttest_1samp(b - a, 0).pvalue, sps.ttest_rel(b, a).pvalue)


class TestMetrics(unittest.TestCase):
    def test_scores(self):
        y, p = [1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1]
        s = scores(y, p, 0.5)
        self.assertEqual((s["tp"], s["fp"], s["fn"], s["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(s["f1"], 50.0)

    def test_ece(self):
        y = np.array([0, 1] * 50)
        self.assertAlmostEqual(ece(y, np.full(100, 0.5)), 0.0)
        self.assertAlmostEqual(ece(y, y.astype(float)), 0.0)
        self.assertAlmostEqual(ece(y, 1 - y.astype(float)), 1.0)


if __name__ == "__main__":
    unittest.main()
