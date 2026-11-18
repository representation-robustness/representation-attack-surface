"""Concurrent jobs share the graph-view cache: exactly one builds it, the others wait and load the same file."""
import multiprocessing as mp
import os
import tempfile
import time
import unittest
from pathlib import Path


def _worker(root, q):
    os.environ["RERUN_ROOT"] = root
    from pipeline.models import graph_views as gv
    gv.ROOT = Path(root)
    built = Path(root) / "builds"

    class FakeVocab:
        def __init__(self, ds):
            with open(built, "a") as f:
                f.write("x")

    def slow_graphs(ds, split, cond):
        time.sleep(1.5)  # long enough for the other process to arrive while this one builds
        return [{"id": i} for i in range(5)]

    gv.Vocab = FakeVocab
    gv.load_graphs = slow_graphs
    gv.to_data = lambda g, *a: g["id"]
    q.put(gv.load_view("toy", "train", "clean", {"level": "ast", "edges": [], "features": "type"}))


class TestViewCacheLock(unittest.TestCase):
    def test_one_build_for_concurrent_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            ps = [ctx.Process(target=_worker, args=(root, q)) for _ in range(3)]
            for p in ps:
                p.start()
            outs = [q.get(timeout=60) for _ in ps]
            for p in ps:
                p.join(60)
            self.assertEqual(outs, [[0, 1, 2, 3, 4]] * 3)
            self.assertEqual((Path(root) / "builds").read_text(), "x")  # built once
            left = [p.name for p in (Path(root) / "cache" / "views" / "toy").iterdir() if ".tmp" in p.name]
            self.assertEqual(left, [])  # no partial files


if __name__ == "__main__":
    unittest.main()
