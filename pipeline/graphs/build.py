"""Build Joern graphs for clean and transformed splits with one pinned Joern.

  python -m pipeline.graphs.build --dataset devign --split test --conditions clean,identifier,...

Each sample becomes one source file <id>.c|.cpp; chunks of CHUNK files are parsed by
joern-parse and exported by export_graphs.sc. Output (gzip JSON lines, one per sample):
  $RERUN_ROOT/data/<ds>/graphs/<split>/<condition>.jsonl.gz
    {"id", "target", "header_normalized", "method", "nodes": [...], "edges": [...]}
Samples Joern cannot turn into a method are retried with the function header reduced
to `int <declarator>` (Joern only; dataset text is untouched). That decision is made on
the clean version and applied to every condition of the sample, so graph differences
between conditions come from the transformation alone. Unchanged transformed samples
reuse the clean graph. Missing samples are listed in graphs/<split>/missing.json.

Resource use: at most --jobs concurrent JVMs (default 2), each -Xmx6g and 2 processors, nice 19 + idle I/O.
"""
import argparse
import gzip
import json
import os
import shutil
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.transforms.core import parse

ROOT = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun"))
JOERN = Path(os.environ.get("RERUN_JOERN", ROOT / "tools/joern-v4.0.655")) / "joern-cli"
JDK = Path(os.environ.get("RERUN_JDK", ROOT / "tools/jdk"))
SCRIPT = Path(__file__).with_name("export_graphs.sc")
WORK = ROOT / "cache/joern_work"
CHUNK = 500
# Joern sizes thread pools to every visible core; cap each JVM to 2 so the whole
# build stays at a few cores on the shared machine.
JVM = ["-J-Xmx6g", "-J-XX:ActiveProcessorCount=2", "-J-XX:ParallelGCThreads=2",
       "-J-XX:CICompilerCount=2", f"-J-Djava.io.tmpdir={ROOT / 'cache/jtmp'}"]
_CHILDREN: set = set()


def _run(cmd, **kw):
    """subprocess.run that registers the child so SIGTERM on us also stops it."""
    proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    _CHILDREN.add(proc)
    try:
        return proc.wait()
    finally:
        _CHILDREN.discard(proc)


def _terminate(signum, frame):
    for proc in list(_CHILDREN):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)
ENV = {**os.environ, "JAVA_HOME": str(JDK),
       "PATH": f"{JDK / 'bin'}:{os.environ['PATH']}"}


def normalize_header(src: str) -> str | None:
    """`static av_cold int f(...)` -> `int f(...)` so Joern's parser finds the method."""
    p = parse(src)
    if p.func is None:
        return None
    decl = p.func.child_by_field_name("declarator")
    if decl is None:
        return None
    b = p.src
    return (b[:p.func.start_byte] + b"int " + b[decl.start_byte:]).decode("utf-8", "replace")


def run_chunk(tag: str, items: list[tuple[str, str]]) -> dict:
    """items: (sample id, source). Returns id -> exported graph record."""
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "src").mkdir(parents=True)
    for sid, src in items:
        ext = ".c" if parse(src).lang == "c" else ".cpp"
        (d / "src" / f"{sid}{ext}").write_text(src)
    nice = ["nice", "-n", "19", "ionice", "-c3"]
    log = open(d / "joern.log", "w")
    _run(nice + [str(JOERN / "joern-parse"), *JVM, "src", "-o", "cpg.bin"],
         cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    _run(nice + [str(JOERN / "joern"), *JVM, "--script", str(SCRIPT),
                 "--param", "cpgFile=cpg.bin", "--param", "outFile=out.jsonl"],
         cwd=d, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    out = {}
    if (d / "out.jsonl").exists():
        for line in open(d / "out.jsonl"):
            r = json.loads(line)
            out[r.pop("file").rsplit(".", 1)[0]] = r
    log.close()
    shutil.copy(d / "joern.log", WORK / f"{tag}.log")
    shutil.rmtree(d)
    return out


def run_all(tag: str, items: list[tuple[str, str]], jobs: int) -> dict:
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    res = {}
    with ThreadPoolExecutor(jobs) as ex:
        for k, r in enumerate(ex.map(lambda a: run_chunk(f"{tag}_{a[0]:04d}", a[1]), enumerate(chunks))):
            res.update(r)
            print(f"  {tag}: chunk {k + 1}/{len(chunks)} ({len(res)} graphs)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", default="clean,identifier,deadcode,controlflow,ren_dead,ren_cf,dead_cf,compound")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _terminate)

    base = ROOT / "data" / args.dataset
    out_dir = base / "graphs" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "cache/jtmp").mkdir(parents=True, exist_ok=True)  # JVM tmpdir; Joern fails silently without it
    conds = args.conditions.split(",")

    def load(cond):
        p = base / ("clean" if cond == "clean" else f"transformed/{args.split}") / \
            (f"{args.split}.jsonl" if cond == "clean" else f"{cond}.jsonl")
        recs = [json.loads(l) for l in open(p)]
        return recs[:args.limit] if args.limit else recs

    clean = load("clean")
    clean_src = {r["id"]: r["func"] for r in clean}
    target = {r["id"]: r["target"] for r in clean}

    # clean pass, then header-normalised retry for samples Joern could not use
    tag = f"{args.dataset}_{args.split}"
    g_clean = run_all(f"{tag}_clean", [(r["id"], r["func"]) for r in clean], args.jobs)
    retry = [(sid, h) for sid in clean_src if sid not in g_clean
             if (h := normalize_header(clean_src[sid])) is not None]
    normalized = {sid for sid, _ in retry}
    g_clean.update(run_all(f"{tag}_clean_hdr", retry, args.jobs))
    # Reference coverage is ~95-99%; a collapse means Joern itself failed (see the chunk logs), not the inputs.
    if len(g_clean) < 0.5 * len(clean_src):
        raise SystemExit(f"only {len(g_clean)}/{len(clean_src)} clean graphs; Joern is failing, see {WORK}/{tag}_clean_0000.log")

    missing = {}
    for cond in conds:
        if cond == "clean":
            graphs = g_clean
        else:
            recs = load(cond)
            todo, graphs = [], {}
            for r in recs:
                if r["func"] == clean_src[r["id"]]:
                    if r["id"] in g_clean:
                        graphs[r["id"]] = g_clean[r["id"]]
                    continue
                src = normalize_header(r["func"]) if r["id"] in normalized else r["func"]
                if src is not None:
                    todo.append((r["id"], src))
            graphs.update(run_all(f"{tag}_{cond}", todo, args.jobs))
        with gzip.open(out_dir / f"{cond}.jsonl.gz", "wt") as f:
            for sid in clean_src:
                if sid in graphs:
                    f.write(json.dumps({"id": sid, "target": target[sid],
                                        "header_normalized": sid in normalized, **graphs[sid]}) + "\n")
        missing[cond] = sorted(set(clean_src) - set(graphs))
        print(f"{cond}: {len(graphs)}/{len(clean_src)} graphs, {len(missing[cond])} missing", flush=True)
    (out_dir / "missing.json").write_text(json.dumps(
        {"header_normalized": sorted(normalized), "missing": missing}, indent=1))


if __name__ == "__main__":
    main()
