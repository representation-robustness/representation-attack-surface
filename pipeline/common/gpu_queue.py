"""File-based GPU job queue that only runs jobs when GPUs are genuinely free.

  python -m pipeline.common.gpu_queue add NAME [--requires JOB,...] -- cmd args...   # enqueue
  python -m pipeline.common.gpu_queue run                          # daemon loop
  python -m pipeline.common.gpu_queue status

Layout: $RERUN_ROOT/runs/queue/{pending,running,done,failed}/<NAME>.json
The daemon wakes every POLL_S seconds and, while fewer than MAX_SLOTS of our GPU
jobs run, launches the oldest pending job through runjob.sh --gpus 1 (which uses
the gpu_guard definition of "free"). A job that finds no free GPU exits 75 and is
returned to pending. Each slot pins its job to its own CPU cores. A job with
`requires` only starts once every named job's manifest records exit code 0.

Packing: with RERUN_JOBS_PER_GPU=k the daemon runs up to RERUN_MAX_GPUS x k jobs (bounded by the slot
core list) and the guard places up to k of them on one GPU, each memory-capped. A packed job that fails
with a CUDA out-of-memory error goes back to pending marked exclusive (whole-GPU budget). Pending jobs
start longest-first by their "est_h" estimate (set by `pipeline.plan enqueue --priority-from REF_ROOT`),
then in the order they were added.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("RERUN_ROOT", Path(__file__).resolve().parents[3] / "rerun"))
REPO = Path(__file__).resolve().parents[2]
Q = ROOT / "runs" / "queue"
POLL_S = int(os.environ.get("RERUN_QUEUE_POLL_S", "120"))
JOBS_PER_GPU = max(1, int(os.environ.get("RERUN_JOBS_PER_GPU", "1")))
MAX_SLOTS = int(os.environ.get("RERUN_MAX_GPUS", "2")) * JOBS_PER_GPU
# Cores each slot is pinned to (3 per slot keeps us small on a shared box); override e.g. RERUN_SLOT_CPUS="0-2 3-5".
# Default: consecutive 3-core groups ending below the last 4 cores (which CPU stages use).
_N = os.cpu_count()
SLOT_CPUS = os.environ.get("RERUN_SLOT_CPUS", " ".join(
    f"{_N - 4 - 3 * (MAX_SLOTS - k)}-{_N - 4 - 3 * (MAX_SLOTS - k) + 2}" for k in range(MAX_SLOTS))).split()
MAX_SLOTS = min(MAX_SLOTS, len(SLOT_CPUS))
SLOT_THREADS = os.environ.get("RERUN_SLOT_THREADS", "3")
EX_TEMPFAIL = 75


def _dirs():
    for d in ("pending", "running", "done", "failed"):
        (Q / d).mkdir(parents=True, exist_ok=True)


def ready(job: dict) -> bool:
    for dep in job.get("requires", []):
        man = ROOT / "runs" / "manifests" / f"{dep}.json"
        if not man.exists() or json.loads(man.read_text()).get("exit_code") != 0:
            return False
    return True


def _oom(name: str) -> bool:
    log = ROOT / "logs" / f"{name}.log"
    if not log.exists():
        return False
    with open(log, "rb") as f:
        f.seek(max(0, log.stat().st_size - 20000))
        tail = f.read().decode("utf-8", "replace")
    return "CUDA out of memory" in tail or "OutOfMemoryError" in tail


def add(name: str, cmd: list[str], requires: list[str] | None = None, est_h: float = 0.0) -> None:
    _dirs()
    for d in ("pending", "running", "done"):
        if (Q / d / f"{name}.json").exists():
            print(f"{name} already {d}; skipping")
            return
    (Q / "failed" / f"{name}.json").unlink(missing_ok=True)
    (Q / "pending" / f"{name}.json").write_text(json.dumps({"name": name, "cmd": cmd, "requires": requires or [], "added": time.time(), "est_h": est_h}))
    print("queued", name)


def status() -> None:
    _dirs()
    for d in ("running", "pending", "failed", "done"):
        names = sorted(p.stem for p in (Q / d).glob("*.json"))
        print(f"{d:8s} {len(names):3d}  {' '.join(names[:12])}{' ...' if len(names) > 12 else ''}")


class _Adopted:
    """A job launched by an earlier daemon: finished when its runjob pid exits; exit code from its manifest."""

    def __init__(self, pid: int, name: str):
        self.pid, self.name = pid, name

    def poll(self):
        if Path(f"/proc/{self.pid}").exists():
            return None
        man = ROOT / "runs" / "manifests" / f"{self.name}.json"
        try:
            return json.loads(man.read_text()).get("exit_code", 1)
        except (OSError, ValueError):
            return 1


def run() -> None:
    _dirs()
    procs: dict[str, tuple] = {}
    locks = ROOT / "runs" / "gpu_locks"
    for stale in (Q / "running").glob("*.json"):
        lk = locks / stale.name
        pid = json.loads(lk.read_text())["pid"] if lk.exists() else None
        if pid and Path(f"/proc/{pid}").exists():  # daemon restarted while this job runs: adopt it, don't rerun it
            used = {sl for _, sl in procs.values()}
            procs[stale.stem] = (_Adopted(pid, stale.stem), next(s for s in range(MAX_SLOTS) if s not in used))
            print(f"{time.strftime('%F %T')} adopted running {stale.stem} (pid {pid})", flush=True)
        else:  # its process is gone: retry it
            stale.rename(Q / "pending" / stale.name)
    backoff: dict[str, float] = {}  # job -> time before which it is not retried (it just found no room)
    while True:
        deferred = False
        for name, (proc, slot) in list(procs.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del procs[name]
            src = Q / "running" / f"{name}.json"
            dst = "pending" if rc == EX_TEMPFAIL else ("done" if rc == 0 else "failed")
            job = json.loads(src.read_text())
            if rc not in (0, EX_TEMPFAIL) and JOBS_PER_GPU > 1 and not job.get("exclusive") and _oom(name):
                job["exclusive"], dst = True, "pending"  # retry with a whole GPU
                src.write_text(json.dumps(job))
            src.rename(Q / dst / f"{name}.json")
            deferred |= rc == EX_TEMPFAIL
            if rc == EX_TEMPFAIL:
                backoff[name] = time.time() + 600
            print(f"{time.strftime('%F %T')} {name} exit={rc} -> {dst}", flush=True)
        free_slots = [s for s in range(MAX_SLOTS) if s not in {sl for _, sl in procs.values()}]
        now = time.time()
        pending = sorted((p for p in Q.joinpath("pending").glob("*.json")
                          if backoff.get(p.stem, 0) <= now and ready(json.loads(p.read_text()))),
                         key=lambda p: (-json.loads(p.read_text()).get("est_h", 0.0), json.loads(p.read_text())["added"]))
        if free_slots and pending:
            job = json.loads(pending[0].read_text())
            slot = free_slots[0]
            pending[0].rename(Q / "running" / pending[0].name)
            env = {**os.environ, "RERUN_CPUS": SLOT_CPUS[slot], "RERUN_THREADS": SLOT_THREADS,
                   "RERUN_EXCLUSIVE": "1" if job.get("exclusive") else "0"}
            proc = subprocess.Popen([str(REPO / "pipeline/common/runjob.sh"), "--gpus", "1", job["name"], "--", *job["cmd"]],
                                    cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs[job["name"]] = (proc, slot)
            print(f"{time.strftime('%F %T')} launched {job['name']} in slot {slot}", flush=True)
            time.sleep(60)  # let it acquire its GPU before considering the next job
            continue
        time.sleep(POLL_S)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "add":
        sep = sys.argv.index("--")
        opts = sys.argv[3:sep]
        req = opts[opts.index("--requires") + 1].split(",") if "--requires" in opts else []
        est = float(opts[opts.index("--est-h") + 1]) if "--est-h" in opts else 0.0
        add(sys.argv[2], sys.argv[sep + 1:], req, est)
    elif cmd == "run":
        run()
    else:
        status()
