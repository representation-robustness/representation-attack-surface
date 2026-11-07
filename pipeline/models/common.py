"""Shared data access and the per-run prediction format.

Every model run writes ONE file:
  $RERUN_ROOT/results/preds/<dataset>/<model>/run_<run>.json.gz
    {"meta": {...}, "threshold": t,
     "valid": {"ids": [...], "y": [...], "p": [...]},
     "test":  {"<condition>": {"ids": [...], "y": [...], "p": [...]}, ...}}
p is P(vulnerable). The threshold is chosen on the validation split (max F1) and
applied unchanged to every test condition. All metrics, significance tests,
attack and transfer analyses are computed from these files only.
"""
import gzip
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RERUN_ROOT", REPO.parent / "rerun"))
CONDITIONS = ["identifier", "deadcode", "controlflow", "ren_dead", "ren_cf", "dead_cf", "compound"]
TEST_KEYS = ["clean"] + CONDITIONS
# Pretrained checkpoints are pinned to exact Hugging Face commits so reruns load identical weights.
HF_REVISIONS = {
    "microsoft/codebert-base": "3b0952feddeffad0063f274080e3c23d75e7eb39",
    "microsoft/graphcodebert-base": "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d",
    "Salesforce/codet5p-220m": "2b92f36e2782341a50551759fdba0dd15e821f99",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cap = os.environ.get("RERUN_GPU_MEM_MIB")  # set by runjob.sh when sharing an idle GPU
        if cap and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, int(cap) * 2**20 / total), 0)
        if torch.cuda.is_available():  # peak memory in the log, for sizing packed GPU jobs
            import atexit
            atexit.register(lambda: print(f"peak_gpu_mem_mib={torch.cuda.max_memory_reserved() // 2**20}", flush=True))
    except ImportError:
        pass


def load_text(dataset: str, split: str, condition: str = "clean") -> list[dict]:
    base = ROOT / "data" / dataset
    p = base / "clean" / f"{split}.jsonl" if condition == "clean" else \
        base / "transformed" / split / f"{condition}.jsonl"
    return [json.loads(l) for l in open(p)]


def load_graphs(dataset: str, split: str, condition: str = "clean"):
    p = ROOT / "data" / dataset / "graphs" / split / f"{condition}.jsonl.gz"
    with gzip.open(p, "rt") as f:
        for line in f:
            yield json.loads(line)


_ABBR = {"identifier": "ren", "deadcode": "dead", "controlflow": "cf"}


def run_name(model: str, train_conditions: list[str]) -> str:
    """Name of a training variant.
      []                                  -> model            (clean training)
      [clean, clean, clean]               -> model_volctl     (clean data repeated: data-volume control)
      [identifier, deadcode, controlflow] -> model_aug        (robustness training, paper Table IX)
      any other subset                    -> model_aug_<fams> (held-out transformation family)
    """
    if not train_conditions:
        return model
    if all(c == "clean" for c in train_conditions):
        return f"{model}_volctl"
    if sorted(train_conditions) == sorted(_ABBR):
        return f"{model}_aug"
    return f"{model}_aug_" + "+".join(_ABBR[c] for c in sorted(train_conditions, key=list(_ABBR).index))


def best_threshold(y, p) -> float:
    """Threshold maximising F1 on (y, p); candidates are the observed scores."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    f1 = 2 * tp / np.maximum(1, tp + fp + ys.sum())
    i = int(np.argmax(f1))
    return float(ps[i])


def git_commit() -> str:
    """HEAD commit, with "+dirty" when tracked files under pipeline/ or configs/ have uncommitted changes."""
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "pipeline", "configs"],
                           capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def save_run(dataset: str, model: str, run: int, meta: dict, valid: dict, test: dict) -> Path:
    thr = best_threshold(valid["y"], valid["p"])
    out = ROOT / "results" / "preds" / dataset / model / f"run_{run}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "dataset": dataset, "model": model, "run": run, "git_commit": git_commit()}

    def pack(d):
        return {"ids": list(d["ids"]), "y": [int(v) for v in d["y"]], "p": [round(float(v), 6) for v in d["p"]]}

    with gzip.open(out, "wt") as f:
        json.dump({"meta": meta, "threshold": thr, "valid": pack(valid),
                   "test": {k: pack(v) for k, v in test.items()}}, f)
    return out
