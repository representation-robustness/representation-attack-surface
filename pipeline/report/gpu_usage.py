"""How efficiently the GPU matrix ran: occupancy, packing and speed versus a reference run.

  python -m pipeline.report.gpu_usage [--root RERUN_ROOT] [--ref REF_ROOT] [--out FILE.md]

Reads only provenance: runs/manifests/*.json (job, GPU, start/end, memory cap, exit code), runs/gpu_locks/
(jobs still running), logs/<job>.log (peak_gpu_mem_mib), logs/gpu_queue.log and logs/deferred.log.
Reports:
  - GPU-hours of finished jobs per model family, and wall-clock span;
  - per GPU: hours with >= 1 of our jobs, mean / max number of our jobs while occupied (packing factor);
  - job-hours / occupied GPU-hours = effective throughput multiplier from packing;
  - speed of each finished job against the same job in --ref (ratio > 1 = slower here), by family;
  - deferrals, out-of-memory retries, failed jobs.
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _t(s):
    return datetime.fromisoformat(s).timestamp()


def jobs(root: Path) -> list[dict]:
    out = []
    for f in (root / "runs" / "manifests").glob("*.json"):
        d = json.loads(f.read_text())
        if not d.get("gpus") or not d.get("start") or not d.get("end"):
            continue
        out.append({"job": d["job"], "gpu": int(str(d["gpus"]).split(",")[0]), "start": _t(d["start"]), "end": _t(d["end"]),
                    "rc": d.get("exit_code"), "cap": d.get("gpu_mem_cap_mib") or None, "running": False})
    now = datetime.now(timezone.utc).timestamp()
    for f in (root / "runs" / "gpu_locks").glob("*.json"):
        d = json.loads(f.read_text())
        out.append({"job": d["job"], "gpu": d["gpus"][0], "start": d["t"], "end": now, "rc": None,
                    "cap": d.get("shared_cap_mib"), "running": True})
    return out


def family(job: str) -> str:
    return re.sub(r"_s\d+$", "", job)


def peak(root: Path, job: str):
    p = root / "logs" / f"{job}.log"
    m = re.findall(r"peak_gpu_mem_mib=(\d+)", p.read_text(errors="replace")[-20000:]) if p.exists() else []
    return int(m[-1]) if m else None


def occupancy(js: list[dict], step: float = 60.0) -> dict:
    """GPU -> (occupied hours, mean jobs while occupied, max jobs)."""
    out = {}
    by = defaultdict(list)
    for j in js:
        by[j["gpu"]].append((j["start"], j["end"]))
    for g, iv in by.items():
        t0, t1 = min(a for a, _ in iv), max(b for _, b in iv)
        ts = np.arange(t0, t1, step)
        n = np.zeros(len(ts))
        for a, b in iv:
            n += (ts >= a) & (ts < b)
        occ = n > 0
        out[g] = (occ.sum() * step / 3600, float(n[occ].mean()) if occ.any() else 0.0, int(n.max()) if len(n) else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from pipeline.models.common import ROOT
    root = a.root or ROOT
    js = jobs(root)
    ok = [j for j in js if j["rc"] == 0]
    L = [f"# GPU usage: `{root}`\n"]
    if not js:
        print("no GPU jobs found")
        return
    span = (max(j["end"] for j in js) - min(j["start"] for j in js)) / 3600
    jh = sum(j["end"] - j["start"] for j in js if j["rc"] in (0, None)) / 3600
    occ = occupancy([j for j in js if j["rc"] in (0, None)])
    gh = sum(v[0] for v in occ.values())
    L.append(f"wall clock {span:.1f} h; finished jobs {len(ok)}, running {sum(j['running'] for j in js)}, "
             f"failed {sum(1 for j in js if j['rc'] not in (0, None, 75))}\n")
    L.append(f"job-hours {jh:.1f} on {gh:.1f} occupied GPU-hours: packing multiplier {jh / max(gh, 1e-9):.2f}x\n")
    L.append("\n| GPU | occupied h | mean jobs while occupied | max jobs |\n|---|---|---|---|")
    for g in sorted(occ):
        L.append(f"| {g} | {occ[g][0]:.1f} | {occ[g][1]:.2f} | {occ[g][2]} |")
    ref = {j["job"]: j for j in jobs(a.ref)} if a.ref else {}
    fam = defaultdict(list)
    for j in ok:
        r = ref.get(j["job"])
        fam[family(j["job"])].append((j["end"] - j["start"], (r["end"] - r["start"]) if r and r["rc"] == 0 else None, peak(root, j["job"])))
    L.append("\n| family | finished | mean h | vs reference (time ratio) | peak GPU MiB |\n|---|---|---|---|---|")
    for k in sorted(fam, key=lambda k: -sum(x[0] for x in fam[k])):
        v = fam[k]
        ratios = [x[0] / x[1] for x in v if x[1]]
        pk = [x[2] for x in v if x[2]]
        L.append(f"| {k} | {len(v)} | {np.mean([x[0] for x in v]) / 3600:.2f} | "
                 f"{np.mean(ratios):.2f}" + (f" (n={len(ratios)})" if ratios else "--") + f" | {max(pk) if pk else '--'} |")
    dl = root / "logs" / "deferred.log"
    n_def = sum(1 for _ in open(dl)) if dl.exists() else 0
    n_oom = sum(1 for p in (root / "logs").glob("*.log") if "out of memory" in p.read_text(errors="replace")[-20000:])
    L.append(f"\ndeferrals (no GPU room at launch): {n_def}; job logs ending in a CUDA OOM (later retried larger): {n_oom}\n")
    text = "\n".join(L) + "\n"
    out = a.out or root / "results" / "gpu_usage.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
