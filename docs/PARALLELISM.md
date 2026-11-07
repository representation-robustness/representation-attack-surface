# How the experiment matrix is parallelized

The full matrix is 265 GPU jobs plus CPU stages. Measured on the reference run, one GPU per job, it costs about
285 GPU-hours. Most of it is a few long jobs:

| Jobs | GPU-hours (reference) |
|---|---|
| Vul-LMGGNN Big-Vul × 5 | 62 |
| CodeT5+ Big-Vul × 5 | 34 |
| ReGVD Big-Vul × 5 | 30 |
| ReGVD Devign controls (aug, volume, held-out) × 25 | 45 |
| Everything else (≈ 225 jobs) | ≈ 115 |

The machine is shared. Our jobs therefore use only GPUs that are free, or whose other users' processes are idle.
A live allowlist keeps them off GPUs that others need. This document describes how jobs are scheduled under
those constraints and how to check that the scheduling was efficient.

## Pieces

| File | Role |
|---|---|
| `pipeline/plan.py` | The matrix: every job, its command, and the data jobs it depends on (`gpu_jobs()`). `enqueue` adds them to the queue. With `RERUN_PRIORITY_FROM=<earlier root>` each job carries the duration it took there (`est_h`). |
| `pipeline/common/gpu_queue.py` | Daemon. Runs up to `RERUN_MAX_GPUS × RERUN_JOBS_PER_GPU` jobs, bounded by the slot core list `RERUN_SLOT_CPUS`. Each loop it launches the ready job with the largest `est_h` (longest-first). A job whose dependencies are not done waits. A job that finds no GPU room exits 75 and backs off 10 min, and the next job is tried. A packed job that runs out of GPU memory is requeued for a larger slot. After a restart, the daemon adopts jobs that are still running. |
| `pipeline/common/gpu_guard.py` | Placement. See below. |
| `pipeline/common/runjob.sh` | Runs one job with `nice 19` and idle I/O, pinned to its slot's cores and thread-capped. It acquires and releases the GPU lock, applies the memory cap (`torch.cuda.set_per_process_memory_fraction`), and writes a log and a provenance manifest. |
| `tools/full_rerun.example.sh` | The unattended driver used for the 2026-11-07 rerun, with the settings used. |
| `pipeline/report/gpu_usage.py` | Efficiency report from manifests and logs (below). |
| `tests/test_gpu_guard.py` | Packing and admission rules on a mocked GPU. |

## Placement rules (`gpu_guard.acquire`)

1. **Eligible GPUs:**
   - free: no process of any user, < 1 GiB used, < 5% utilization over 30 s;
   - otherwise idle-shared: other users' processes stayed below 5% utilization for 2 continuous minutes, and at least 20 GiB is free;
   - always restricted to `RERUN_GPUS` or the file `$RERUN_ROOT/runs/gpu_allowed`, which is re-read on every call;
   - at most `RERUN_MAX_GPUS` distinct GPUs.
2. **Fixed budget.** When we first take a GPU, its budget is fixed:
   - free GPU: total − used − 2 GiB;
   - idle-shared GPU: free − 6 GiB, so the other user keeps headroom to resume.

   The other users' memory at that moment is recorded.
3. **Packing.** Up to `RERUN_JOBS_PER_GPU` of our jobs share a GPU we hold.
   - **Cap per job:** 1.25 × the job's measured peak memory + 1 GiB. The peak is the largest `peak_gpu_mem_mib` logged by earlier runs of the same model, and `runs/mem_estimates.json` covers models not yet measured. An unmeasured job gets half the budget.
   - **Admission:** a job is admitted only while the caps on that GPU fit in the budget, so our total never exceeds the budget.
   - **Best fit:** large holes are kept for large jobs.
   - **No growth by others:** nothing more is added once the other users' memory on that GPU has grown.
4. **Out of memory.** A job that ran out of memory under a packed cap is retried with ≥ 1.4 × its old cap: in a packed slot if one fits, otherwise alone on a GPU.

## Why the jobs pack well

The long jobs are of two kinds.

- **GPU-bound:** CodeT5+ and CodeBERT, ~13–21 GB, near 100% utilization when alone.
- **CPU-bound:** ReGVD's upstream code builds every batch's token graph in NumPy on the CPU and then runs a small
  GNN in float64. It needs ~3 GB and keeps the GPU at ~20% when alone. This is the authors' design and is left
  unchanged for fidelity. The graph models (REVEAL, ECG RGCN, ANGLE, VulGNN, Devign) are small: 0.04–0.7 h per run.

Packing several CPU-bound or small jobs next to one GPU-bound job raises GPU-hours of work per occupied GPU-hour.
The cost is that packed jobs share the GPU and their pinned cores, so each runs somewhat slower.

## Not parallelized, by choice

- **One GPU per job.** Every job's training loop is single-device (no DDP), as in each detector's reference
  implementation. Seeds and detectors parallelize across jobs instead.
- **CPU stages are capped at 4 cores.** Joern graph building is the long one (~4.5 h). This is a shared-server
  limit, not a technical one. GPU jobs that only need text (transformers, ReGVD) start as soon as the transforms
  exist and do not wait for graphs.
- **Graph builds run in the order of the GPU work they unblock** (`GRAPH_ORDER` in `pipeline/plan.py`). A dataset's
  graph models need all three of its splits, so the builds are dataset-major: Big-Vul first, because it gates the
  five Vul-LMGGNN Big-Vul runs, the longest chain (12 h each). The earlier split-major order (all test splits,
  then all valid, then all train) left Big-Vul train for last.

## Speed-ups deliberately not applied

The rerun's purpose is to check that the same experiments reproduce the same values. Speed-ups that change what a
model computes, or that would make the rerun differ from the reference run, are therefore out of scope.

| Possible speed-up | GPU-hours | Wall-clock effect | Why not applied |
|---|---|---|---|
| Cache ReGVD's per-sample token graphs across epochs. Upstream rebuilds them on the CPU in every forward pass. | Not measured; a large share of ReGVD's ~87 reference GPU-hours | Small: ReGVD jobs already run packed beside GPU-bound jobs and are not on the longest chain | Modifies the upstream model's forward. Results would be identical, but the finished ReGVD runs would then differ in code from the rest. A candidate for future runs. |
| Mixed precision for Vul-LMGGNN (fp32 in the reference run) | Not measured; Vul-LMGGNN totals ~87 reference GPU-hours | Up to ~half a day: Vul-LMGGNN Big-Vul is the longest chain | Changes the numerics, so a difference from the reference could not be separated from the precision change |
| Larger batches or fewer epochs | large | large | Changes the training protocol (Table I) |
| Multi-GPU data parallelism within a job | none | Shorter single jobs only when GPUs are idle | No idle GPUs here: packing already keeps all allowed GPUs busy |
| More cores for the Joern graph builds | none | ~1 h earlier start for the graph models | The other cores belong to other users on this shared server |

CodeBERT and CodeT5+ already train with mixed precision (`torch.cuda.amp`), in both runs.

## Lower bound on wall-clock time

After the graphs are built, the GPU-bound jobs (CodeT5+, CodeBERT, Vul-LMGGNN) need ~160 reference GPU-hours. Each
keeps a GPU near 100%, so on G GPUs the run cannot finish in less than about 160 / G hours: ~40 h on 4 GPUs.
The CPU-bound and small jobs (~85 GPU-hours) are packed next to them and add little wall-clock time. The only
ways below that bound change the experiment (mixed precision, larger batches) or need more GPUs; neither was used.

## Checking efficiency

```bash
RERUN_ROOT=<root> python -m pipeline.report.gpu_usage --ref <earlier root>
```

| Quantity | Meaning |
|---|---|
| Packing multiplier | Job-hours ÷ occupied GPU-hours. |
| Occupancy per GPU | Hours occupied, mean and max of our jobs on it. |
| Time ratio per model family | Each job's duration against the same job in the reference run. > 1 means slower here, for example from packing or from sharing cores. |
| Counts | Deferrals, out-of-memory retries and failures. |

The report's inputs, per-job manifests and logs (start/end, GPU, memory cap, peak memory), are in
`$RERUN_ROOT/runs/manifests/` and `$RERUN_ROOT/logs/`.

**Snapshot, 2026-11-07 14:40, 13.6 h into the rerun** (full report: `docs/gpu_usage_2026-11-07.md`) (4 GPUs available at most, 2 for most of the time):
- job-hours / occupied GPU-hours = **2.02×**;
- GPU 3 averaged 3.1 of our jobs while occupied (four ReGVD runs);
- ReGVD Big-Vul ran at 1.24× its solo time while packed four to a GPU, so ~3.1 / 1.24 ≈ 2.5× more ReGVD work per GPU-hour;
- CodeT5+ Big-Vul ran at 1.01× its solo time.

## Changes to the scheduler during the 2026-11-07 rerun

The run started with 2 jobs per GPU, each capped at half the budget. The changes below were made while it ran.
The daemon was restarted for them, and running jobs were adopted, not restarted. Model and training code did not
change.

- An out-of-memory job asked for 1.5× its cap, a few MiB too much for a packed slot, so it took a whole GPU.
  The factor is now 1.4×.
- The daemon slept 10 min after any deferral. It now polls every 2 min and backs off only the deferred job,
  which had been blocking the jobs behind it.
- Fixed half-budget caps wasted memory on ~3 GB ReGVD jobs. Caps are now set from measured peaks, with up to 4 jobs per GPU.
- The allowlist file was added so that GPUs 0–2 could be released to another user without a restart.
- The remaining graph builds were reordered to Big-Vul train first. This became the plan's default.
- The driver's final "wait for the queue to empty" check counted the empty folders themselves and never ended.
  It is fixed in `tools/full_rerun.example.sh`, which also runs the comparisons and this report at the end.
- **Incident:** an in-place edit of `runjob.sh` while jobs ran broke the final step of three running jobs (bash
  reads scripts incrementally). Their outputs were complete, and their manifests were restored and marked
  `"recovered"`. Never edit `runjob.sh` while jobs run.
