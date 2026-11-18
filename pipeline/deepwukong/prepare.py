"""Transform the SARD source files behind DeepWukong's test XFGs, function by function.

  python -m pipeline.deepwukong.prepare --condition deadcode [--workers 4]

Every source file referenced by a shipped test XFG is parsed with tree-sitter. Each function definition is
transformed with pipeline.transforms (the same code and validity gate as the main study) and spliced back
only if it passes the gate; otherwise it stays original and is recorded as such. Output, per condition:
  work/<cond>/data/CWE119/source-code/<file>   transformed file (bytes outside edited functions untouched)
  work/<cond>/linemap.json                     {file: [new line or null for each original line]}
  work/<cond>/functions.jsonl                  per function: original line span, changed, valid, reason
"""
import argparse
import difflib
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from pipeline.deepwukong.common import ROOT, SHIPPED, test_index, work

_IDENT = re.compile(r"[A-Za-z_]\w*")
LANG = {"c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp"}


def _norm(line: str) -> str:
    return re.sub(r"\s+", "", _IDENT.sub("I", line))


def _map_block(old: list[str], new: list[str]) -> list[int | None]:
    """Original line index -> new line index within one function (identifier-insensitive alignment)."""
    a, b = [_norm(x) for x in old], [_norm(x) for x in new]
    out: list[int | None] = [None] * len(old)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = j1 + k
        elif tag == "replace":
            for i in range(i1, i2):
                best = max(range(j1, j2), key=lambda j: difflib.SequenceMatcher(None, a[i], b[j]).ratio())
                if difflib.SequenceMatcher(None, a[i], b[best]).ratio() >= 0.6:
                    out[i] = best
    return out


def _functions(root):
    """Function definitions with a compound body, outermost only, in source order."""
    out, stack = [], [root]
    while stack:
        x = stack.pop()
        if x.type == "function_definition":
            body = x.child_by_field_name("body")
            if body is not None and body.type == "compound_statement":
                out.append(x)
                continue
        stack.extend(reversed(x.children))
    return sorted(out, key=lambda n: n.start_byte)


def process(job):
    rel, cond = job
    from pipeline.transforms.core import _PARSERS, apply_condition
    from pipeline.transforms.validate import check
    raw = (SHIPPED / "source-code" / rel).read_bytes().decode("latin-1")  # byte-preserving
    lang = LANG.get(rel.rsplit(".", 1)[-1].lower(), "c")
    b = raw.encode("utf-8")
    funcs = _functions(_PARSERS[lang].parse(b).root_node)
    pieces, recs, news, pos = [], [], [], 0
    for f in funcs:
        old = b[f.start_byte:f.end_byte].decode("utf-8")
        rec = {"file": rel, "start": f.start_point[0] + 1, "end": f.end_point[0] + 1}
        new = old
        if cond != "clean":
            cand, infos = apply_condition(old, cond)
            inv = {}
            for info in infos:
                inv.update({v: k for k, v in (info.pop("mapping", None) or {}).items()})
            v = check(old, cand, lang, rename_inverse=inv)
            rec.update(changed=v["changed"], valid=v["valid"], reason=v.get("reject_reason"))
            if v["valid"]:
                new = cand
        pieces += [b[pos:f.start_byte], new.encode("utf-8")]
        pos = f.end_byte
        recs.append(rec)
        news.append(new)
    pieces.append(b[pos:])
    out_text = b"".join(pieces).decode("utf-8")
    # line map: identity shift outside functions, identifier-insensitive alignment inside
    o_lines, n_lines = raw.split("\n"), out_text.split("\n")
    lmap: list[int | None] = [None] * len(o_lines)
    shift, prev = 0, 0
    for rec, new in zip(recs, news):
        s, e = rec["start"] - 1, rec["end"] - 1
        for r in range(prev, s):
            lmap[r] = r + shift
        span_new = new.count("\n") + 1
        for k, j in enumerate(_map_block(o_lines[s:e + 1], n_lines[s + shift:s + shift + span_new])):
            lmap[s + k] = None if j is None else s + shift + j
        shift += span_new - (e - s + 1)
        prev = e + 1
    for r in range(prev, len(o_lines)):
        lmap[r] = r + shift
    assert len(n_lines) == len(o_lines) + shift, (rel, len(n_lines), len(o_lines), shift)
    dst = work(cond) / "data" / "CWE119" / "source-code" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out_text.encode("latin-1"))  # inverse of the byte-preserving decode
    return rel, [None if j is None else j + 1 for j in lmap], recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted({r["file"] for r in test_index()})
    w = work(a.condition)
    w.mkdir(parents=True, exist_ok=True)
    linemap = {}
    with open(w / "functions.jsonl", "w") as fj, ProcessPoolExecutor(a.workers) as ex:
        for i, (rel, lm, recs) in enumerate(ex.map(process, [(f, a.condition) for f in files], chunksize=4)):
            linemap[rel] = lm
            for r in recs:
                fj.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                print(f"{i}/{len(files)} files", flush=True)
    (w / "linemap.json").write_text(json.dumps(linemap))
    print(f"{a.condition}: {len(files)} files written to {w}")


if __name__ == "__main__":
    main()
