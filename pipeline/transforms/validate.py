"""Per-sample validity gate for transformed functions.

A transformed sample is VALID iff
  1. a function definition is found before and after,
  2. it actually changed (whitespace-insensitive),
  3. re-parsing introduces no new tree-sitter ERROR/MISSING nodes, and
  4. gcc/g++ -fsyntax-only reports no *new* error diagnostics.

Benchmark functions are compiled without their project headers, so the
original already produces errors (unknown types/macros). Check 4 is therefore
differential: diagnostics are normalised (quoted identifiers and positions
removed) and compared as multisets; any error kind/count present in the
transformed function but not the original makes the sample invalid. This is
conservative: error-recovery noise can only reject valid samples, never
accept broken ones that introduce a new error kind.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections import Counter

from pipeline.transforms.core import count_errors, parse

COMPILERS = {
    "c": ["gcc", "-fsyntax-only", "-std=gnu11", "-w", "-fmax-errors=0", "-x", "c"],
    "cpp": ["g++", "-fsyntax-only", "-std=gnu++17", "-w", "-fpermissive", "-fmax-errors=0", "-x", "c++"],
}
_QUOTED = re.compile(r"(‘[^’]*’|'[^']*'|\"[^\"]*\")")
_ERR = re.compile(r"^[^:]*:\d+:\d+: (?:fatal )?error: (.*)$")
_SUGGEST = re.compile(r";\s*did you mean <Q>\?$")  # gcc spelling hints depend on local names
# "x undeclared" diagnostics are compared as sets of names (mapped back through the
# rename table), because g++ may repeat them when a statement gains a new scope.
_UNDECL = re.compile(r"^[‘'](.+?)[’'] (?:was not declared in this scope|undeclared)")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def compile_errors(src: str, lang: str, timeout: int = 30) -> tuple[Counter, set] | None:
    """(multiset of normalised non-undeclared errors, set of undeclared names); None on timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".c" if lang == "c" else ".cpp", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run(COMPILERS[lang] + [path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(path)
    errs, undecl = Counter(), set()
    for line in r.stderr.splitlines():
        m = _ERR.match(line)
        if not m:
            continue
        u = _UNDECL.match(m.group(1))
        if u:
            undecl.add(u.group(1))
        else:
            errs[_SUGGEST.sub("", _QUOTED.sub("<Q>", m.group(1)).strip())] += 1
    return errs, undecl


def check(original: str, transformed: str, lang: str, orig_cache: dict | None = None,
          rename_inverse: dict | None = None) -> dict:
    """Validity record for one (original, transformed) pair.

    orig_cache lets callers reuse the original's parse/compile results across conditions;
    rename_inverse maps fresh names back to originals (for undeclared-name comparison).
    """
    oc = orig_cache if orig_cache is not None else {}
    if "parse" not in oc:
        po = parse(original, lang)
        oc["parse"] = (po.func is not None, count_errors(po.root))
        oc["compile"] = compile_errors(original, lang)
    o_func, o_perr = oc["parse"]
    o_cerr = oc["compile"]

    rec = {"lang": lang, "func_found": o_func, "changed": _norm_ws(original) != _norm_ws(transformed)}
    pt = parse(transformed, lang)
    rec["func_found_after"] = pt.func is not None
    rec["new_parse_errors"] = max(0, count_errors(pt.root) - o_perr)
    t_cerr = compile_errors(transformed, lang) if rec["changed"] else o_cerr
    if o_cerr is None or t_cerr is None:
        rec["new_compile_errors"], rec["compile_timeout"] = None, True
    else:
        inv = rename_inverse or {}
        new = t_cerr[0] - o_cerr[0]
        new_undecl = {inv.get(n, n) for n in t_cerr[1]} - o_cerr[1]
        rec["new_compile_errors"] = sum(new.values()) + len(new_undecl)
        rec["new_compile_error_kinds"] = sorted(new)[:5] + [f"undeclared {n}" for n in sorted(new_undecl)[:5]]
        rec["orig_compile_errors"] = sum(o_cerr[0].values()) + len(o_cerr[1])
    rec["valid"] = bool(o_func and rec["func_found_after"] and rec["changed"]
                        and rec["new_parse_errors"] == 0 and rec["new_compile_errors"] == 0)
    if not rec["valid"]:
        rec["reject_reason"] = (
            "no_function" if not o_func else
            "unchanged" if not rec["changed"] else
            "function_lost" if not rec["func_found_after"] else
            "new_parse_errors" if rec["new_parse_errors"] else
            "compile_timeout" if rec.get("compile_timeout") else
            "new_compile_errors")
    return rec
