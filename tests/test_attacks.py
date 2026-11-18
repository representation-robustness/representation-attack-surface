"""Attack definitions on a hand-built example (no files needed)."""
import unittest
from unittest import mock

from pipeline.eval import attacks


def fake_validity(valid_conds):
    def v(dataset, cond):
        cost = {"identifiers_renamed": 1, "occurrences_renamed": 1, "sites": 1, "tokens_added": 5 if "dead" in cond else 0}
        return {"x": (cond in valid_conds, cost)}
    return v


class TestMinimalSequence(unittest.TestCase):
    def run_with(self, evading, valid):
        run = {"pred": {c: {"x": 0 if c in evading else 1} for c in attacks.CONDITIONS}}
        with mock.patch.object(attacks, "validity", fake_validity(valid)):
            return attacks.minimal_sequence("d", run, "x")

    def test_lowest_budget_wins(self):
        best = self.run_with({"compound", "ren_dead"}, set(attacks.CONDITIONS))
        self.assertEqual(best[1], "ren_dead")
        self.assertEqual(best[0][0], 2)

    def test_tie_broken_by_fewest_tokens(self):
        best = self.run_with({"deadcode", "controlflow"}, set(attacks.CONDITIONS))
        self.assertEqual(best[1], "controlflow")

    def test_invalid_transform_never_counts(self):
        self.assertIsNone(self.run_with({"identifier"}, set(attacks.CONDITIONS) - {"identifier"}))


class TestPairing(unittest.TestCase):
    def test_pairs(self):
        runs = {"a": [7, 42, 100], "b": [7, 42, 100], "lr": list(range(30))}
        with mock.patch.object(attacks, "runs_of", lambda d, m: runs[m]):
            self.assertEqual(attacks.paired_runs("d", "a", "b"), [(7, 7), (42, 42), (100, 100)])
            self.assertEqual(attacks.paired_runs("d", "a", "lr"), [(7, 0), (42, 1), (100, 2)])


if __name__ == "__main__":
    unittest.main()
