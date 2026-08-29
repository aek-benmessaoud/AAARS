#!/usr/bin/env python
"""lambda_sweep_launcher.py — run the AAARS risk_lambda sensitivity sweep
across 6 parallel workers (bounded pool) and merge results.

Cells: lam in {0,0.5,1,2,4} x alloc {boustro,minerich} x seed-half {10:40,50:40}
= 20 jobs; up to 6 run concurrently.
"""
import subprocess
import sys
import os
import json
import time

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "lambda_sweep.py")
RAW = os.path.join(ROOT, "results", "raw")

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0]
ALLOCS = ["boustro", "minerich"]
SEED_HALVES = [(10, 40), (50, 40)]
MAX_CONCURRENT = 6

jobs = []
for lam in LAMS:
    for alloc in ALLOCS:
        for off, n in SEED_HALVES:
            out = f"lam_{lam}_{alloc}_{off}.json"
            jobs.append({"lam": lam, "alloc": alloc, "off": off, "n": n,
                         "out": out})

print(f"total jobs: {len(jobs)}", flush=True)

active = []
idx = 0


def launch(job):
    global active
    out_abs = os.path.join(RAW, job["out"])
    for suf in ("", ".log", ".err.log"):
        p = out_abs + suf
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, SCRIPT, "--lam", str(job["lam"]),
           "--alloc", job["alloc"], "--offset", str(job["off"]),
           "--seeds", str(job["n"]), "--out", job["out"]]
    with open(out_abs + ".log", "w") as logf, \
         open(out_abs + ".err.log", "w") as errf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf)
    active.append({"job": job, "proc": p})
    print(f"start lam={job['lam']} alloc={job['alloc']} off={job['off']} "
          f"pid={p.pid}", flush=True)


t0 = time.perf_counter()
while idx < len(jobs) or active:
    while idx < len(jobs) and len(active) < MAX_CONCURRENT:
        launch(jobs[idx])
        idx += 1
    # poll
    done = [a for a in active if a["proc"].poll() is not None]
    for a in done:
        print(f"done lam={a['job']['lam']} alloc={a['job']['alloc']} "
              f"off={a['job']['off']} rc={a['proc'].returncode} "
              f"[{(time.perf_counter()-t0)/60:.1f}m]", flush=True)
        active.remove(a)
    if not active:
        break
    time.sleep(5)

print(f"all workers finished in {(time.perf_counter()-t0)/60:.1f} min",
      flush=True)

# Merge
merged = []
for job in jobs:
    p = os.path.join(RAW, job["out"])
    if os.path.exists(p):
        with open(p) as f:
            merged += json.load(f)
        print(f"chunk {job['out']}: {job['n']} records", flush=True)
    else:
        print(f"WARNING missing {job['out']}", flush=True)

final = os.path.join(RAW, "lambda_sweep.json")
with open(final, "w") as f:
    json.dump(merged, f, indent=2, default=str)
print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
