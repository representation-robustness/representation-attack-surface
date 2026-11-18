"""Executable semantic-equivalence tests for the v3 transforms.

Each case is a complete C program whose function `target` exercises a
construct that broke or could break v2. For every condition we transform only
`target`, compile original and transformed with gcc, run both over many
inputs, and require byte-identical output.

Usage: python -m pipeline.transforms.semantic_tests [--keep DIR]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.transforms.core import CONDITIONS, apply_condition, parse

DRIVER = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
static int g_calls;
static int side(int x) { g_calls++; printf("side(%d) ", x); return x; }
static void note(const char *s) { printf("[%s] ", s); }
static int twice(int x) { return 2 * x; }
#define for_each_upto(i, n) for ((i) = 0; (i) < (n); (i)++)
%TARGET%
int main(void) {
    for (int i = -6; i <= 12; i++) {
        g_calls = 0;
        %CALL%
        printf(" calls=%d\n", g_calls);
    }
    return 0;
}
"""

CASES = {
    # opaque predicate must use the int param, never the pointer
    "pointer_and_int_params": (
        """int target(char *buf, int n) {
    int s = 0;
    for (int i = 0; i < n && i < 8; i++) s += buf[i];
    note("loop done");
    if (n > 3) s += 1; else s -= 1;
    while (n-- > 0) { s += 2; }
    return s;
}""", 'char b[16] = "abcdefghijklmno"; printf("%d", target(b, i));'),
    # braceless if/else nested in braceless for: dangling-else hazard
    "dangling_else": (
        """int target(int n) {
    int a = 0;
    for (int i = 0; i < 5; i++)
        if (i < n) a += side(i); else a -= 1;
    if (n > 0)
        if (n > 5) note("big"); else note("small");
    else
        note("neg");
    return a;
}""", 'printf("%d", target(i));'),
    # if/else and calls inside switch cases
    "switch_cases": (
        """int target(int x) {
    int r = 0;
    switch (x % 4) {
    case 0: if (x > 4) r = 1; else r = 2; break;
    case 1: note("one"); r = 3;
    case 2: r += side(x); break;
    default: { r = -1; note("def"); }
    }
    return r;
}""", 'printf("%d", target(i));'),
    # loops with break/continue: flattening and wrapping must not capture them
    "break_continue": (
        """int target(int n) {
    int s = 0, k = 0;
    while (k < 20) {
        k++;
        if (k == n) continue;
        if (k > n + 5) break;
        s += k;
        s *= 1;
        s -= 0;
    }
    side(s); note("x"); side(k);
    return s;
}""", 'printf("%d", target(i));'),
    # goto + labels
    "goto_labels": (
        """int target(int n) {
    int r = 0;
    if (n < 0) goto fail;
    r = n * 2;
    if (r > 10) { r = 10; } else { r += 1; }
    note("ok");
    return r;
fail:
    note("fail");
    return -1;
}""", 'printf("%d", target(i));'),
    # local function pointer called through its (renamed) name
    "local_function_pointer": (
        """int target(int n) {
    int (*fp)(int) = twice;
    int v = fp(n);
    if (v > 4) v = fp(v); else v = side(v);
    return v;
}""", 'printf("%d", target(i));'),
    # shadowing in nested scopes, static local
    "shadowing_static": (
        """int target(int n) {
    static int calls = 0;
    int x = n;
    calls++;
    { int x = n + 100; side(x); }
    for (int x = 0; x < 2; x++) side(x);
    return x + (calls > 0);
}""", 'printf("%d", target(i));'),
    # sequential side effects: order must be preserved by the dispatcher
    "sequence_order": (
        """int target(int n) {
    int a = 0;
    a = side(n);
    a += side(n + 1);
    a *= side(2);
    a -= side(3);
    note("end");
    return a;
}""", 'printf("%d", target(i));'),
    # no integer parameter at all -> literal predicates
    "no_int_param": (
        """int target(const char *s) {
    int n = 0;
    while (*s) { if (*s == 'a') n++; else n += 2; s++; }
    note("done");
    return n;
}""", 'printf("%d", target(i % 2 ? "banana" : "cab"));'),
    # volatile and unsigned/size_t params
    "volatile_and_unsigned": (
        """unsigned target(volatile int v, size_t len, unsigned char c) {
    unsigned acc = 0;
    while (len > 0) { acc += c; len--; }
    if (v) acc++; else acc--;
    note("u");
    return acc;
}""", 'printf("%u", target(i, (size_t)(i < 0 ? 0 : i), (unsigned char)(i + 3)));'),
    # lowercase loop macro heading a block and a single statement (Linux style)
    "loop_macro": (
        """int target(int n) {
    int i, s = 0;
    for_each_upto(i, n) {
        s += side(i);
        note("in");
    }
    for_each_upto(i, 3)
        side(i);
    note("after");
    if (s > 3) s = 1; else s = 2;
    return s;
}""", 'printf("%d", target(i));'),
    # do-while, ternaries, comma operator in condition
    "do_while_comma": (
        """int target(int n) {
    int i = 0, s = 0;
    do { s += i; i++; } while (i < 3);
    while ((s++, s < n)) { side(s); }
    if (n & 1) s = n > 2 ? s : -s; else s = 0;
    return s;
}""", 'printf("%d", target(i));'),
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)


def build_and_run(program: str, work: Path, tag: str) -> str:
    c, exe = work / f"{tag}.c", work / tag
    c.write_text(program)
    r = run(["gcc", "-std=gnu11", "-O0", "-w", "-o", str(exe), str(c)])
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[:2000]}")
    return run([str(exe)]).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, help="keep generated programs here")
    args = ap.parse_args()
    scratch = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun")) / "runs"
    scratch.mkdir(parents=True, exist_ok=True)
    work = args.keep or Path(tempfile.mkdtemp(prefix="semtest_", dir=scratch))
    work.mkdir(parents=True, exist_ok=True)

    fails = 0
    for name, (target, call) in CASES.items():
        assert parse(target).func is not None, name
        base = DRIVER.replace("%CALL%", call)
        ref = build_and_run(base.replace("%TARGET%", target), work, f"{name}__orig")
        for cond in CONDITIONS:
            new, infos = apply_condition(target, cond)
            changed = new != target
            try:
                out = build_and_run(base.replace("%TARGET%", new), work, f"{name}__{cond}")
                ok = out == ref
            except RuntimeError as e:
                ok, out = False, str(e)
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails += 1
            print(f"{status} {name:24s} {cond:12s} changed={changed} "
                  f"sites={[i.get('n_sites', i.get('n_occurrences')) for i in infos]}")
            if not ok:
                print("---- transformed ----\n" + new + "\n---- output/diag ----\n" + out[:1500])
    print(f"\n{fails} failures over {len(CASES) * len(CONDITIONS)} case-conditions; programs in {work}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
