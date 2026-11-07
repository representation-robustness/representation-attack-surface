"""GPU packing rules, on a mocked GPU (nothing touches a real GPU)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.common import gpu_guard as g


class TestPacking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = [mock.patch.object(g, "LOCK_DIR", Path(self.tmp.name) / "gpu_locks"), mock.patch.object(g, "JOBS_PER_GPU", 2),
                  mock.patch.object(g, "MAX_OURS", 1), mock.patch.object(g, "ALLOWED", None),
                  mock.patch.object(g, "free_gpus", lambda: []),
                  mock.patch.object(g, "gpu_snapshot", lambda: {6: {"bus": "b6", "used": 4766, "total": 49140, "util": 0, "has_procs": True}}),
                  mock.patch.object(g, "idle_shared_gpus", lambda excl: {6: 38230} if 6 not in excl else {})]
        for x in self.p:
            x.start()
        self.others = {6: 4766}
        self.p.append(mock.patch.object(g, "others_mib", lambda: dict(self.others)))
        self.p[-1].start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self.tmp.cleanup()

    def test_two_jobs_share_budget_third_defers(self):
        a = g.acquire(1, "a", os.getpid())
        b = g.acquire(1, "b", os.getpid())
        self.assertEqual(a, ([6], 38230 // 2))
        self.assertEqual(b, ([6], 38230 // 2))
        self.assertIsNone(g.acquire(1, "c", os.getpid()))

    def test_no_packing_when_other_user_grows(self):
        g.acquire(1, "a", os.getpid())
        self.others[6] += 8000
        self.assertIsNone(g.acquire(1, "b", os.getpid()))

    def test_oom_retry_needs_bigger_slot(self):
        import json
        (Path(self.tmp.name) / "manifests").mkdir(exist_ok=True)
        (Path(self.tmp.name) / "manifests" / "big.json").write_text(json.dumps({"gpu_mem_cap_mib": "12761"}))
        # 1.4 x 12761 = 17865 <= 38230 // 2 = 19115: it fits a packed slot, so another job can still join
        self.assertEqual(g.acquire(1, "big", os.getpid(), exclusive=True), ([6], 19115))
        self.assertEqual(g.acquire(1, "b", os.getpid()), ([6], 19115))
        (Path(self.tmp.name) / "manifests" / "huge.json").write_text(json.dumps({"gpu_mem_cap_mib": "19115"}))
        g.release("b")
        self.assertIsNone(g.acquire(1, "huge", os.getpid(), exclusive=True))  # 26.8 GB > any slot or free budget

    def test_memory_aware_packing(self):
        needs = {"t5": 18784, "r1": 3196, "r2": 3196, "r3": 3196, "r4": 3196}
        with mock.patch.object(g, "JOBS_PER_GPU", 4), mock.patch.object(g, "need_mib", lambda j: needs.get(j)):
            self.assertEqual(g.acquire(1, "t5", os.getpid()), ([6], int(18784 * 1.25) + 1024))
            for j in ("r1", "r2"):  # caps 24504 + 2 x 5019 = 34542 <= 38230
                self.assertEqual(g.acquire(1, j, os.getpid()), ([6], int(3196 * 1.25) + 1024))
            self.assertIsNone(g.acquire(1, "r3", os.getpid()))  # a third would exceed the budget (MAX_OURS = 1 GPU)

    def test_exclusive(self):
        self.assertEqual(g.acquire(1, "a", os.getpid(), exclusive=True), ([6], 38230))
        self.assertIsNone(g.acquire(1, "b", os.getpid()))


if __name__ == "__main__":
    unittest.main()
