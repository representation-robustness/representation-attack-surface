"""Shared-machine GPU guard.

A GPU is "free" only if, over a sampling window, it hosts no compute process
of ANY user, uses < MAX_USED_MIB memory, and stays < MAX_UTIL % utilization.
Free GPUs are always preferred. Only when none is free, a GPU whose other-user
processes are IDLE (< MAX_UTIL % over a longer SHARED_SAMPLES window) and that
has >= SHARED_MIN_FREE_MIB free memory may be shared: the job is then hard-capped
to the free memory minus SHARED_HEADROOM_MIB (RERUN_GPU_MEM_MIB, applied in
pipeline.models.common.set_seed), so the other user keeps room to resume.
Our own jobs may hold at most MAX_OURS GPUs in total (RERUN_MAX_GPUS, default 2), tracked with lock
files; RERUN_GPUS="0,1" (or the file $RERUN_ROOT/runs/gpu_allowed, read on every call) restricts the guard to those GPU ids; RERUN_SHARE_IDLE=0 disables sharing.
If nothing qualifies we defer; we never share a GPU that is computing.

Packing (RERUN_JOBS_PER_GPU=k, default 1): up to k of OUR jobs may share one GPU that we already hold.
When we first take a GPU we fix its memory budget (free GPU: total - 2 GiB; idle-shared GPU: free -
SHARED_HEADROOM_MIB). Each job is capped at 1.25 x its measured peak + 1 GiB (peak_gpu_mem_mib from earlier
runs of the same model; half the budget if unmeasured) and admitted only while the caps on that GPU fit in
the budget, so our total never exceeds it.
A further job is added only while the other users' memory on that GPU has not grown since the budget was
fixed (i.e. their idle jobs have not resumed). A job started with --exclusive (RERUN_EXCLUSIVE=1) needs a
GPU with none of our jobs and gets the whole budget.

CLI:
  python -m pipeline.common.gpu_guard acquire --n 1 --job NAME   # prints "0,6", "3;CAP_MIB" (shared) or exits 75
  python -m pipeline.common.gpu_guard release --job NAME
  python -m pipeline.common.gpu_guard status
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK_DIR = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun")) / "runs" / "gpu_locks"
MAX_OURS = int(os.environ.get("RERUN_MAX_GPUS", "2"))          # GPUs our jobs may hold at once
_ALLOW_FILE = LOCK_DIR.parent / "gpu_allowed"   # optional: "3,4,5,6"; read on every call, so it can be changed live
_allow = os.environ.get("RERUN_GPUS") or (_ALLOW_FILE.read_text().strip() if _ALLOW_FILE.exists() else "")
ALLOWED = {int(g) for g in _allow.replace(" ", "").split(",") if g} or None
SHARE_IDLE = os.environ.get("RERUN_SHARE_IDLE", "1") == "1"    # 0: never share a GPU with another user's process
MAX_USED_MIB = 1024
MAX_UTIL = 5
SAMPLES = 6
SAMPLE_GAP_S = 5
SHARED_SAMPLES = 12          # x SHARED_GAP_S = 2 minutes of continuous idleness
SHARED_GAP_S = 10
SHARED_MIN_FREE_MIB = 20 * 1024
SHARED_HEADROOM_MIB = 6 * 1024
JOBS_PER_GPU = max(1, int(os.environ.get("RERUN_JOBS_PER_GPU", "1")))
FREE_HEADROOM_MIB = 2 * 1024
EX_TEMPFAIL = 75


def _smi(query: str, kind: str) -> list:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_snapshot() -> dict:
    gpus = {int(i): {"bus": b.lower(), "used": int(u), "total": int(t), "util": int(ut)}
            for i, b, u, t, ut in _smi("index,pci.bus_id,memory.used,memory.total,utilization.gpu", "gpu")}
    busy = set()
    for bus, _pid in _smi("gpu_bus_id,pid", "compute-apps"):
        for i, g in gpus.items():
            if g["bus"] == bus.lower():
                busy.add(i)
    for i in gpus:
        gpus[i]["has_procs"] = i in busy
    return gpus


def _live_locks(full: bool = False) -> dict:
    """job -> gpu ids (or the whole lock record), dropping locks whose owning pid is gone."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    held = {}
    for f in LOCK_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        if Path(f"/proc/{d['pid']}").exists():
            held[d["job"]] = d if full else d["gpus"]
        else:
            f.unlink()
    return held


def others_mib() -> dict:
    """GPU index -> memory (MiB) held by processes of OTHER users."""
    snap = gpu_snapshot()
    by_bus = {g["bus"]: i for i, g in snap.items()}
    me = os.getuid()
    out = {i: 0 for i in snap}
    for bus, pid, mem in _smi("gpu_bus_id,pid,used_memory", "compute-apps"):
        try:
            uid = next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith("Uid:"))
        except (OSError, StopIteration, ValueError):
            uid = -1  # unknown owner counts as another user
        if uid != me and bus.lower() in by_bus:
            out[by_bus[bus.lower()]] += int(mem) if mem.isdigit() else 0
    return out


def _prev_cap(job: str) -> int | None:
    """Memory cap of this job's previous (failed) attempt, from its manifest."""
    man = LOCK_DIR.parent / "manifests" / f"{job}.json"
    try:
        cap = json.loads(man.read_text()).get("gpu_mem_cap_mib")
        return int(cap) if cap else None
    except (OSError, ValueError):
        return None


def need_mib(job: str) -> int | None:
    """Expected peak GPU memory of a job: the largest `peak_gpu_mem_mib` logged by finished runs of the same
    model and dataset (any seed), else of the same model on any dataset, in this root."""
    import re as _re
    logs = LOCK_DIR.parent.parent / "logs"
    stem = _re.sub(r"_s\d+$", "", job)
    model = _re.sub(r"_(devign|diversevul|bigvul)$", "", stem)
    for pat in (f"{stem}_s*.log", f"{model}_*_s*.log"):
        peaks = []
        for f in logs.glob(pat):
            if _re.sub(r"_(devign|diversevul|bigvul)_s\d+$", "", f.stem) != model:
                continue
            try:
                tail = f.read_text(errors="replace")[-20000:]
            except OSError:
                continue
            if "out of memory" not in tail:  # an OOM run's peak is its cap, not its need
                peaks += [int(x) for x in _re.findall(r"peak_gpu_mem_mib=(\d+)", tail)]
        if peaks:
            return max(peaks)
    est = LOCK_DIR.parent / "mem_estimates.json"  # optional {"<model>_<dataset>" or "<model>": MiB}, e.g. from nvidia-smi
    if est.exists():
        e = json.loads(est.read_text())
        return e.get(stem) or e.get(model)
    return None


def _cap_for(job: str, budget: int) -> int:
    """Memory cap for a packed job: 1.25 x its measured peak + 1 GiB, or half the GPU budget if never measured."""
    need = need_mib(job)
    return min(budget, int(need * 1.25) + 1024) if need else budget // 2


def _pack(locks: dict, job: str, min_cap: int = 0) -> tuple[int, int] | None:
    """A GPU we already hold with room for this job within its fixed budget: (gpu, cap_mib)."""
    per = {}
    for d in locks.values():
        if len(d["gpus"]) == 1 and d.get("budget_mib"):
            per.setdefault(d["gpus"][0], []).append(d)
    if not per:
        return None
    now = others_mib()
    best = None
    for g, ds in per.items():
        budget = ds[0]["budget_mib"]
        cap = max(_cap_for(job, budget), min_cap)
        room = budget - sum(d.get("shared_cap_mib") or 0 for d in ds)
        if (len(ds) >= JOBS_PER_GPU or any(d.get("exclusive") for d in ds) or (ALLOWED is not None and g not in ALLOWED)
                or now.get(g, 0) > max(d.get("others_mib", 0) for d in ds) + 256 or cap > room):
            continue
        if best is None or room - cap < best[2]:  # best fit: keep large holes for large jobs
            best = (g, cap, room - cap)
    return best[:2] if best else None


def free_gpus() -> list:
    ok = None
    for k in range(SAMPLES):
        snap = gpu_snapshot()
        now = {i for i, g in snap.items()
               if not g["has_procs"] and g["used"] < MAX_USED_MIB and g["util"] < MAX_UTIL}
        ok = now if ok is None else ok & now
        if k < SAMPLES - 1:
            time.sleep(SAMPLE_GAP_S)
    return sorted(ok)


def idle_shared_gpus(exclude: set) -> dict:
    """GPU -> memory cap (MiB) for GPUs whose other-user processes stayed idle the whole window."""
    caps = None
    for k in range(SHARED_SAMPLES):
        snap = gpu_snapshot()
        now = {i: g["total"] - g["used"] - SHARED_HEADROOM_MIB for i, g in snap.items()
               if i not in exclude and g["util"] < MAX_UTIL and g["total"] - g["used"] >= SHARED_MIN_FREE_MIB}
        caps = now if caps is None else {i: min(c, now[i]) for i, c in caps.items() if i in now}
        if not caps:
            return {}
        if k < SHARED_SAMPLES - 1:
            time.sleep(SHARED_GAP_S)
    return caps


def acquire(n: int, job: str, owner_pid: int, exclusive: bool = False) -> tuple[list, int | None] | None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_DIR / ".mutex", "w") as m:
        fcntl.flock(m, fcntl.LOCK_EX)
        locks = _live_locks(full=True)
        ours = {g for d in locks.values() for g in d["gpus"]}
        # A job retried after running out of memory (exclusive) first tries a packed slot with >= 1.4x its old cap.
        min_cap = 0
        if exclusive and JOBS_PER_GPU > 1 and (prev := _prev_cap(job)):
            min_cap, exclusive = int(prev * 1.4), False
        packing = JOBS_PER_GPU > 1 and n == 1 and not exclusive
        budget, others = None, 0
        if packing and (got := _pack(locks, job, min_cap)):
            pick, cap = [got[0]], got[1]
            budget = next(d["budget_mib"] for d in locks.values() if d["gpus"] == pick and d.get("budget_mib"))
            others = max(d.get("others_mib", 0) for d in locks.values() if d["gpus"] == pick)
        else:
            if len(ours) + n > MAX_OURS:
                return None
            cand = [g for g in free_gpus() if g not in ours and (ALLOWED is None or g in ALLOWED)]
            excl = ours | ({g for g in gpu_snapshot()} - ALLOWED if ALLOWED is not None else set())
            cap = None
            if len(cand) >= n:
                pick = cand[:n]
                if n == 1 and JOBS_PER_GPU > 1:
                    snap = gpu_snapshot()[pick[0]]
                    budget = snap["total"] - snap["used"] - FREE_HEADROOM_MIB
                    others = others_mib()[pick[0]]
                    cap = max(_cap_for(job, budget), min_cap)
                    exclusive = exclusive or cap > budget
                    cap = budget if exclusive else cap
            elif n == 1 and SHARE_IDLE and (shared := idle_shared_gpus(excl)):
                g = max(shared, key=shared.get)  # most free memory
                pick, budget, others = [g], shared[g], others_mib()[g]
                if budget < min_cap:
                    return None
                cap = max(_cap_for(job, budget), min_cap)
                exclusive = exclusive or cap > budget
                cap = budget if (exclusive or JOBS_PER_GPU == 1) else cap
            else:
                return None
        (LOCK_DIR / f"{job}.json").write_text(
            json.dumps({"job": job, "pid": owner_pid, "gpus": pick, "shared_cap_mib": cap, "budget_mib": budget,
                        "others_mib": others, "exclusive": exclusive, "t": time.time()}))
        return pick, cap


def release(job: str) -> None:
    (LOCK_DIR / f"{job}.json").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--n", type=int, default=1)
    a.add_argument("--job", required=True)
    a.add_argument("--pid", type=int, default=os.getppid(),
                   help="pid whose lifetime owns the lock (default: calling shell)")
    a.add_argument("--exclusive", action="store_true", help="never pack another of our jobs on this GPU")
    r = sub.add_parser("release")
    r.add_argument("--job", required=True)
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "acquire":
        got = acquire(args.n, args.job, args.pid, args.exclusive)
        if got is None:
            print("no free or idle-shareable GPU (or our 2-GPU cap reached); deferring", file=sys.stderr)
            sys.exit(EX_TEMPFAIL)
        pick, cap = got
        print(",".join(map(str, pick)) + (f";{cap}" if cap is not None else ""))
    elif args.cmd == "release":
        release(args.job)
    else:
        snap = gpu_snapshot()
        held = _live_locks(full=True)
        for i, g in snap.items():
            print(f"gpu{i} used={g['used']}/{g['total']}MiB util={g['util']}% procs={g['has_procs']}")
        print("ours:", held or "none")


if __name__ == "__main__":
    main()
