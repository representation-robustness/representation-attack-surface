"""Structural invariants of the transforms, plus the executable semantic tests when gcc is present."""
import re
import shutil
import subprocess
import sys
import unittest

from pipeline.transforms.core import CONDITIONS, apply_condition, parse

SAMPLES = [
    """int process(int len, char *buf) {
    int count = 0;
    for (int i = 0; i < len; i++)
        count += buf[i];
    if (count > 10) { count = 10; } else count--;
    return count;
}""",
    """static void f(struct s *p, unsigned n) {
    size_t k;
    while (n--) { p->x += n; k = n; }
    switch (k) { case 1: g(p); break; default: break; }
}""",
]
TOK = re.compile(r"[A-Za-z_]\w*|\d+|\S")


class TestTransforms(unittest.TestCase):
    def test_outputs_parse_without_errors(self):
        for s in SAMPLES:
            for c in CONDITIONS:
                new, _ = apply_condition(s, c)
                self.assertEqual(parse(new).errors, 0, f"{c}:\n{new}")

    def test_renaming_only_changes_identifiers(self):
        for s in SAMPLES:
            new, _ = apply_condition(s, "identifier")
            a, b = TOK.findall(s), TOK.findall(new)
            self.assertEqual(len(a), len(b))
            mapping = {}
            for x, y in zip(a, b):
                if x != y:
                    self.assertRegex(y, r"^__v_\d{4}$")
                    self.assertEqual(mapping.setdefault(x, y), y, "mapping must be consistent")
            self.assertEqual(len(set(mapping.values())), len(mapping), "mapping must be one-to-one")
            for kept in ("process", "f", "g", "x"):  # function and field names are never renamed
                self.assertNotIn(kept, mapping)

    def test_dead_code_keeps_original_tokens_in_order(self):
        for s in SAMPLES:
            new, _ = apply_condition(s, "deadcode")
            it = iter(TOK.findall(new))
            self.assertTrue(all(t in it for t in TOK.findall(s)), "original tokens must be a subsequence")

    def test_deterministic(self):
        for s in SAMPLES:
            for c in CONDITIONS:
                self.assertEqual(apply_condition(s, c)[0], apply_condition(s, c)[0])

    @unittest.skipUnless(shutil.which("gcc"), "gcc not available")
    def test_executable_semantic_tests(self):
        r = subprocess.run([sys.executable, "-m", "pipeline.transforms.semantic_tests"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:])


if __name__ == "__main__":
    unittest.main()
