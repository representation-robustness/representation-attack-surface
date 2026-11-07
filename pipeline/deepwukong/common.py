"""Shared paths and helpers for the DeepWukong case study (SARD CWE-119, upstream preprocessed data).

Layout under $RERUN_ROOT/data/deepwukong/:
  raw/Data.7z                       authors' archive (manual download; sha256 in DATA_7Z_SHA256)
  upstream_data/CWE119/             extracted: source-code/ (+ manifest.xml), XFG/, train/val/test.json, w2v.wv
  work/<condition>/                 per condition: data/CWE119/source-code (test files only), csv/, XFG/,
                                    linemap.json, validity.jsonl, index.jsonl
DeepWukong code runs from the pinned upstream checkout ($RERUN_UPSTREAM/DeepWukong) in its own venv;
its scripts resolve data/... and ts_logger/... relative to the working directory, so every step chdirs.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun"))
UPSTREAM = Path(os.environ.get("RERUN_UPSTREAM", ROOT / "upstream")) / "DeepWukong"
JDK11 = Path(os.environ.get("RERUN_JDK11", ROOT / "tools/jdk11"))
BASE = ROOT / "data" / "deepwukong"
SHIPPED = BASE / "upstream_data" / "CWE119"
DATA_7Z_SHA256 = "d0cc5ceabff06ece94ad3bb728241f5b9b2bb7f9cf37949cf542695325b5c869"
CONDITIONS = ["clean", "identifier", "deadcode", "controlflow", "ren_dead", "ren_cf", "dead_cf", "compound"]
KINDS = ["call", "array", "ptr", "arith"]
JOERN_CP = ":".join(str(UPSTREAM / "joern" / p) for p in [
    "projects/extensions/joern-fuzzyc/build/libs/joern-fuzzyc.jar",
    "projects/extensions/jpanlib/build/libs/jpanlib.jar",
    "projects/octopus/lib/*"])


def work(cond: str) -> Path:
    return BASE / "work" / cond


def use_upstream(cwd: Path) -> None:
    """Import DeepWukong modules (they read data/sensiAPI.txt relative to the working directory)."""
    (cwd / "data").mkdir(parents=True, exist_ok=True)
    api = cwd / "data" / "sensiAPI.txt"
    if not api.exists():
        api.write_text((UPSTREAM / "data" / "sensiAPI.txt").read_text())
    os.chdir(cwd)
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def test_index() -> list[dict]:
    """The shipped (deduplicated) test XFGs: id, testcase, kind, key line, source file (relative), label."""
    import pickle
    cache = BASE / "test_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    for p in json.loads((SHIPPED / "test.json").read_text()):
        _, _, _, tc, kind, name = p.split("/")
        x = pickle.load(open(SHIPPED / "XFG" / tc / kind / name, "rb"))
        rel = x.graph["file_paths"][0].split("source-code/", 1)[1]
        key = int(name.split(".")[0])
        out.append({"id": f"{tc}/{kind}/{key}", "testcase": tc, "kind": kind, "key_line": key,
                    "file": rel, "label": int(x.graph["label"])})
    cache.write_text(json.dumps(out))
    return out


def manifest_flaws() -> dict:
    """(testcase id, relative file) -> set of flaw/mixed lines, exactly as upstream getCodeIDtoPathDict."""
    import xml.etree.ElementTree as ET
    res = {}
    for t in ET.parse(SHIPPED / "source-code" / "manifest.xml").getroot().findall("testcase"):
        for f in t.findall("file"):
            res[(t.attrib["id"], f.attrib["path"])] = {int(x.attrib["line"]) for x in f.findall("flaw") + f.findall("mixed")}
    return res
