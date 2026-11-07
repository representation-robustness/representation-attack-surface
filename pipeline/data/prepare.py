"""Canonical clean splits with stable sample ids.

  devign     : CodeXGLUE function.json + canonical train/valid/test index files
  bigvul     : released splits (dataset_release/in_repo/bigvul/splits)
  diversevul : released balanced splits (dataset_release/in_repo/diversevul_dataset/splits)

Writes $RERUN_ROOT/data/<dataset>/clean/{train,valid,test}.jsonl with records
{"id", "func", "target"} and a manifest with input/output sha256 and counts.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun"))
RELEASE = Path(os.environ.get("DATASET_RELEASE", Path(__file__).resolve().parents[3] / "dataset_release")) / "in_repo"
UPSTREAM = Path(os.environ.get("RERUN_UPSTREAM", ROOT / "upstream"))
CODEXGLUE = UPSTREAM / "CodeXGLUE/Code-Code/Defect-detection/dataset"
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(dataset: str, splits: dict, inputs: list[Path]) -> None:
    out = ROOT / "data" / dataset / "clean"
    out.mkdir(parents=True, exist_ok=True)
    man = {"dataset": dataset, "inputs": {str(p): sha256(p) for p in inputs}, "splits": {}}
    for s, recs in splits.items():
        p = out / f"{s}.jsonl"
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        man["splits"][s] = {"n": len(recs), "n_vul": sum(r["target"] for r in recs), "sha256": sha256(p)}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(dataset, {s: (v["n"], v["n_vul"]) for s, v in man["splits"].items()})


def devign():
    src = ROOT / "data/devign/raw/function.json"
    funcs = json.load(open(src))
    splits = {}
    for s in SPLITS:
        idx = [int(l) for l in open(CODEXGLUE / f"{s}.txt") if l.strip()]
        splits[s] = [{"id": f"devign_{i:05d}", "func": funcs[i]["func"], "target": int(funcs[i]["target"]),
                      "project": funcs[i]["project"], "commit_id": funcs[i]["commit_id"]} for i in idx]
    write("devign", splits, [src] + [CODEXGLUE / f"{s}.txt" for s in SPLITS])


def released(dataset: str, subdir: str):
    d = RELEASE / subdir / "splits"
    splits = {}
    for s in SPLITS:
        recs = []
        for i, line in enumerate(open(d / f"{s}.jsonl")):
            r = json.loads(line)
            recs.append({"id": f"{dataset}_{s}_{i:06d}", "func": r["func"], "target": int(r["target"])})
        splits[s] = recs
    write(dataset, splits, [d / f"{s}.jsonl" for s in SPLITS])


if __name__ == "__main__":
    devign()
    released("bigvul", "bigvul")
    released("diversevul", "diversevul_dataset")
