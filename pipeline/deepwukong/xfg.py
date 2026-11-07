"""Build DeepWukong XFGs for one condition with the upstream code, paired to the shipped test XFGs.

  python -m pipeline.deepwukong.xfg --condition deadcode [--jobs 2]

1. Joern (DeepWukong's bundled fuzzy C parser, JDK 11) parses the condition's source tree -> csv/.
2. For every test file: upstream build_PDG / build_XFG with the manifest flaw lines mapped through the line map.
3. Each shipped test XFG (testcase, kind, key line) is paired with the XFG whose key line is the mapped key
   line. It is symbolized exactly as upstream (clean_gadget + tokenize_code_line) and saved under its
   original id. graph["label"] keeps the clean ground-truth label (the transformations preserve semantics);
   the label DeepWukong's labeller would now assign is kept as graph["label_recomputed"].
Instrumentation per test XFG goes to work/<cond>/index.jsonl: seed found or lost, key line mapped, valid
scope (key line inside a validly transformed function), slice nodes/edges, slice lines mapped back to the
original, lines without an original (inserted code), recomputed label. Per file: key lines per kind.
"""
import argparse
import json
import os
import pickle
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pipeline.deepwukong.common import JDK11, JOERN_CP, KINDS, manifest_flaws, test_index, use_upstream, work


def run_joern(w, dirs, jobs):
    env = {**os.environ, "PATH": f"{JDK11 / 'bin'}:{os.environ['PATH']}"}
    batches = [dirs[i:i + 150] for i in range(0, len(dirs), 150)]

    def one(k):
        out = w / "csv" / f"b{k:03d}"
        if (out / ".done").exists():
            return 0
        rc = subprocess.run(["java", "-Xmx3g", "-XX:ActiveProcessorCount=2", "-cp", JOERN_CP, "tools.parser.ParserMain",
                             "-outformat", "csv", "-outdir", str(out), *batches[k]], env=env,
                            capture_output=True, text=True).returncode
        if rc == 0:
            (out / ".done").write_text("")
        return rc
    with ThreadPoolExecutor(jobs) as ex:
        rcs = list(ex.map(one, range(len(batches))))
    assert not any(rcs), f"joern failed in batches {[k for k, r in enumerate(rcs) if r]}"
    loc = {}
    for k, b in enumerate(batches):
        for d in b:
            loc[d] = w / "csv" / f"b{k:03d}"
    return loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    w = work(a.condition)
    src_root = w / "data" / "CWE119" / "source-code"
    idx = test_index()
    flaws = manifest_flaws()
    linemap = json.loads((w / "linemap.json").read_text())
    funcs = defaultdict(list)
    for line in open(w / "functions.jsonl"):
        r = json.loads(line)
        funcs[r["file"]].append(r)

    files = sorted({r["file"] for r in idx})
    dirs = sorted({os.path.realpath(src_root / os.path.dirname(f)) for f in files})
    loc = run_joern(w, dirs, a.jobs)

    use_upstream(w)  # cwd = w, so data/CWE119/source-code/... resolves as upstream expects
    from src.data_generator import build_PDG, build_XFG
    from src.preprocess.symbolizer import clean_gadget, tokenize_code_line

    by_file = defaultdict(list)
    for r in idx:
        by_file[(r["testcase"], r["file"])].append(r)
    out_xfg = w / "XFG"
    paths, file_stats = [], {}
    with open(w / "index.jsonl", "w") as fo:
        for (tc, rel), items in sorted(by_file.items()):
            lm = linemap[rel]
            inv = {j: i + 1 for i, j in enumerate(lm) if j is not None}
            vul = {lm[L - 1] for L in flaws.get((tc, rel), set()) if 0 < L <= len(lm) and lm[L - 1] is not None}
            src_rel = f"data/CWE119/source-code/{rel}"
            real = os.path.realpath(src_root / rel)
            PDG, keys = build_PDG(str(loc[os.path.dirname(real)]) + real, "data/sensiAPI.txt", src_rel)
            res = build_XFG(PDG, keys, vul) or {k: [] for k in KINDS}
            file_stats[f"{tc}|{rel}"] = {k: len(res[k]) for k in KINDS}
            lines = open(src_rel, encoding="utf-8", errors="ignore").readlines()
            for r in items:
                key = lm[r["key_line"] - 1] if r["key_line"] <= len(lm) else None
                fn = next((f for f in funcs[rel] if f["start"] <= r["key_line"] <= f["end"]), None)
                rec = {"id": r["id"], "label": r["label"], "key_mapped": key is not None,
                       "in_function": fn is not None, "valid_scope": bool(fn and fn.get("valid", a.condition == "clean"))}
                g = next((x for x in res[r["kind"]] if x.graph["key_line"] == key), None) if key else None
                rec["found"] = g is not None
                if g is not None:
                    sym = clean_gadget([lines[n - 1] for n in g])
                    toks = {n: tokenize_code_line(s, False) for n, s in zip(g, sym)}
                    for n, t in toks.items():
                        g.nodes[n]["code_sym_token"] = t
                    g.remove_nodes_from([n for n, t in toks.items() if not t])
                    rec["found"] = g.number_of_nodes() > 0
                if rec["found"]:
                    rec.update(nodes=g.number_of_nodes(), edges=g.number_of_edges(),
                               label_recomputed=int(g.graph["label"]),
                               slice_orig=sorted(inv[n] for n in g if n in inv),
                               inserted_lines=sum(n not in inv for n in g))
                    g.graph["label_recomputed"] = g.graph["label"]
                    g.graph["label"] = r["label"]
                    g.graph["file_paths"] = [str(w / src_rel)]
                    p = out_xfg / r["testcase"] / r["kind"] / f"{r['key_line']}.xfg.pkl"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "wb") as f:
                        pickle.dump(g, f)
                    paths.append({"id": r["id"], "path": str(p)})
                fo.write(json.dumps(rec) + "\n")
    (w / "xfg_paths.json").write_text(json.dumps(paths))
    (w / "file_stats.json").write_text(json.dumps(file_stats))
    n = len(idx)
    print(f"{a.condition}: {len(paths)}/{n} test XFGs paired ({100 * len(paths) / n:.1f}%)")


if __name__ == "__main__":
    main()
