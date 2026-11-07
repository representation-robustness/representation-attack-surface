"""The complete experiment plan: every job the paper's tables need, in dependency order.

  python -m pipeline.plan list [STAGE]      # print job names and commands
  python -m pipeline.plan run STAGE         # run a CPU stage's jobs in order via runjob.sh (skips jobs already done)
  python -m pipeline.plan enqueue           # add every GPU job to the polite GPU queue (idempotent)

Stages: fetch, data, transforms, graphs, cpu_models, gpu_models, results.
A job counts as done when its manifest ($RERUN_ROOT/runs/manifests/<job>.json) records exit code 0,
so a stage can be re-run after an interruption and continues where it stopped.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

from pipeline.models.common import REPO, ROOT

PY = os.environ.get("RERUN_PY", sys.executable)
DATASETS = ["devign", "diversevul", "bigvul"]
SPLITS = ["test", "valid", "train"]
ALL_CONDS = "clean,identifier,deadcode,controlflow,ren_dead,ren_cf,dead_cf,compound"
SEEDS = [42, 7, 100, 999, 1337]
FAMILIES = ["identifier", "deadcode", "controlflow"]
# Graph conditions per split: test needs every condition; training augmentation (Devign only) needs the families.
GRAPH_ORDER = ["bigvul", "devign", "diversevul"]
GRAPH_CONDS = {("devign", "train"): "clean," + ",".join(FAMILIES)}
GRAPH_MODELS = ["reveal", "reveal_struct", "devign_ggnn", "ecg_rgcn", "ecg_rgcn_struct", "angle", "angle_struct",
                "vulgnn", "vulgnn_struct", "vulgnn_struct_cb"]
GRAPH_ONLY_ON = {"devign_ggnn": ["devign"], "vulgnn_struct_cb": ["devign"]}
# Devign-only training-data controls (rebuttal): all families, clean x3 volume control, each family held out.
CONTROLS = {"aug": ",".join(FAMILIES), "volctl": "clean,clean,clean",
            **{"ho_" + "_".join(f for f in FAMILIES if f != h): ",".join(f for f in FAMILIES if f != h) for h in FAMILIES}}
CONTROL_MODELS = ["codebert", "regvd", "reveal"]
DEVIGN_FUNCTION_JSON = ("1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF",
                        "0a3b2d561dc6280e53795886ede727d0045c016d083905ba3e9ce384a7eab246")


def m(module, *args):
    return [PY, "-m", module, *map(str, args)]


def cpu_jobs():
    """(stage, name, cmd, env) in run order."""
    jobs = [("fetch", "fetch_devign", m("pipeline.plan", "fetch-devign"), {})]
    jobs.append(("data", "prepare_clean", m("pipeline.data.prepare"), {}))
    # Transformed test sets for every dataset; Devign train/valid too (training-data controls).
    for ds, sp in [(ds, "test") for ds in DATASETS] + [("devign", "train"), ("devign", "valid")]:
        jobs.append(("transforms", f"transform_{ds}_{sp}",
                     m("pipeline.transforms.build", "--dataset", ds, "--split", sp, "--workers", 4), {}))
    jobs.append(("transforms", "semantic_tests", m("pipeline.transforms.semantic_tests"), {}))
    jobs.append(("transforms", "audit_sample", m("pipeline.transforms.audit_sample"), {}))
    # Dataset-major, in order of the GPU work each dataset's graphs unblock (a dataset's graph models need all three
    # of its splits): Big-Vul first (Vul-LMGGNN Big-Vul, the longest jobs, ~62 GPU-h), then Devign, then DiverseVul.
    for ds in GRAPH_ORDER:
        for sp in SPLITS:
            conds = ALL_CONDS if sp == "test" else GRAPH_CONDS.get((ds, sp), "clean")
            jobs.append(("graphs", f"graphs_{ds}_{sp}",
                         m("pipeline.graphs.build", "--dataset", ds, "--split", sp, "--conditions", conds, "--jobs", 2), {}))
    for ds in DATASETS:
        jobs.append(("cpu_models", f"tfidf_{ds}", m("pipeline.models.tfidf_lr", "--dataset", ds), {}))
        jobs.append(("cpu_models", f"cpglr_{ds}", m("pipeline.models.cpg_lr", "--dataset", ds), {}))
    jobs.append(("results", "refresh", m("pipeline.report.refresh"), {}))
    return jobs


def gpu_jobs():
    """(name, cmd, requires) for every neural run."""
    out = []
    text = lambda ds: ["prepare_clean", f"transform_{ds}_test"]  # noqa: E731
    graphs = lambda ds: text(ds) + [f"graphs_{ds}_{sp}" for sp in SPLITS]  # noqa: E731
    for seed in SEEDS:
        for ds in DATASETS:
            for model in ("codebert", "codet5plus"):
                out.append((f"{model}_{ds}_s{seed}", m("pipeline.models.transformer", "--model", model, "--dataset", ds,
                                                       "--seed", seed), text(ds)))
            out.append((f"regvd_{ds}_s{seed}", m("pipeline.models.regvd", "--dataset", ds, "--seed", seed), text(ds)))
            out.append((f"lmggnn_{ds}_s{seed}", m("pipeline.models.lmggnn", "--dataset", ds, "--seed", seed), graphs(ds)))
            for model in GRAPH_MODELS:
                if ds in GRAPH_ONLY_ON.get(model, DATASETS):
                    out.append((f"{model}_{ds}_s{seed}", m("pipeline.models.graph_train", "--model", model, "--dataset", ds,
                                                           "--seed", seed), graphs(ds)))
        for tag, conds in CONTROLS.items():
            for model in CONTROL_MODELS:
                name = f"{model}_{tag}_devign_s{seed}"
                if model == "codebert":
                    cmd = m("pipeline.models.transformer", "--model", "codebert")
                elif model == "regvd":
                    cmd = m("pipeline.models.regvd")
                else:
                    cmd = m("pipeline.models.graph_train", "--model", "reveal")
                req = graphs("devign") if model == "reveal" else text("devign")
                req = req + ["transform_devign_train", "transform_devign_valid"]
                out.append((name, cmd + ["--dataset", "devign", "--seed", str(seed), "--train-conditions", conds], req))
    return out


def done(name):
    p = ROOT / "runs" / "manifests" / f"{name}.json"
    return p.exists() and json.loads(p.read_text()).get("exit_code") == 0


def run_stage(stage):
    for st, name, cmd, env in cpu_jobs():
        if st != stage:
            continue
        if done(name):
            print(f"[skip] {name} already done")
            continue
        print(f"[run ] {name}", flush=True)
        rc = subprocess.run([str(REPO / "pipeline/common/runjob.sh"), name, "--", *cmd],
                            cwd=REPO, env={**os.environ, **env}).returncode
        if rc != 0:
            sys.exit(f"{name} failed (rc={rc}); see {ROOT}/logs/{name}.log")


def selected(name):
    """Optional subset of the GPU matrix, e.g. to split it across machines:
    RERUN_SEEDS="42,7"  RERUN_DATASETS="devign"  RERUN_JOBS="^(codebert|regvd)_"  (regex on the job name)."""
    seeds, dsets, rx = (os.environ.get(k) for k in ("RERUN_SEEDS", "RERUN_DATASETS", "RERUN_JOBS"))
    return ((not seeds or any(name.endswith(f"_s{s}") for s in seeds.split(","))) and
            (not dsets or any(f"_{d}_s" in name for d in dsets.split(","))) and
            (not rx or re.search(rx, name) is not None))


def est_hours(ref) -> dict:
    """job -> hours it took in a reference run (its manifests), used to start the longest jobs first."""
    from datetime import datetime
    out = {}
    for f in (ref / "runs" / "manifests").glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("exit_code") == 0 and d.get("start") and d.get("end"):
            out[d["job"]] = (datetime.fromisoformat(d["end"]) - datetime.fromisoformat(d["start"])).total_seconds() / 3600
    return out


def enqueue():
    """Queue every GPU job. RERUN_PRIORITY_FROM=<another RERUN_ROOT> orders them longest-first by that run's durations."""
    from pathlib import Path
    from pipeline.common import gpu_queue
    ref = os.environ.get("RERUN_PRIORITY_FROM")
    est = est_hours(Path(ref)) if ref else {}
    for name, cmd, req in gpu_jobs():
        if selected(name) and not done(name):
            gpu_queue.add(name, cmd, req, est.get(name, 0.0))


def fetch_devign():
    """Devign function.json (the CodeXGLUE source file) from its public Google Drive id, checksum-verified."""
    import gdown
    fid, sha = DEVIGN_FUNCTION_JSON
    dst = ROOT / "data/devign/raw/function.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        gdown.download(id=fid, output=str(dst), quiet=False)
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"{dst}: sha256 {got} != expected {sha}")
    print("function.json ok", sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "run", "enqueue", "fetch-devign"])
    ap.add_argument("stage", nargs="?")
    a = ap.parse_args()
    if a.action == "list":
        for st, name, cmd, _ in cpu_jobs():
            if a.stage in (None, st):
                print(f"{st:11s} {name:26s} {'done' if done(name) else '    '}  {' '.join(cmd[1:])}")
        if a.stage in (None, "gpu_models"):
            for name, cmd, req in gpu_jobs():
                print(f"{'gpu_models':11s} {name:26s} {'done' if done(name) else '    '}  {' '.join(cmd[1:])}")
    elif a.action == "run":
        run_stage(a.stage)
    elif a.action == "enqueue":
        enqueue()
    else:
        fetch_devign()


if __name__ == "__main__":
    main()
