#!/usr/bin/env python
"""lambda_sweep_resume.py — run the missing lambda-sweep jobs (6 concurrent)
and merge all chunks (existing + new) into lambda_sweep.json."""
import subprocess
import sys
import os
import json
import time

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "lambda_sweep.py")
RAW = os.path.join(ROOT, "results", "raw")

# Only the still-missing jobs
NEED = [
    (1.0, "minerich", 50, 40, "lam_1.0_minerich_50.json"),
    (2.0, "minerich", 10, 40, "lam_2.0_minerich_10.json"),
    (2.0, "minerich", 50, 40, "lam_2.0_minerich_50.json"),
    (4.0, "boustro", 10, 40, "lam_4.0_boustro_10.json"),
    (4.0, "boustro", 50, 40, "lam_4.0_boustro_50.json"),
    (4.0, "minerich", 10, 40, "lam_4.0_minerich_10.json"),
    (4.0, "minerich", 50, 40, "lam_4.0_minerich_50.json"),
]
MAX_CONCURRENT = 6

# Full job grid (for merge completeness)
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0]
ALLOCS = ["boustro", "minerich"]
HALVES = [(10, 40), (50, 40)]
ALL_JOBS = [f"lam_{lam}_{a}_{off}.json"
            for lam in LAMS for a in ALLOCS for off, _ in HALVES]

active = []
idx = 0
t0 = time.perf_counter()


def launch(job):
    lam, a, off, n, out = job
    out_abs = os.path.join(RAW, out)
    for suf in ("", ".log", ".err.log"):
        p = out_abs + suf
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, SCRIPT, "--lam", str(lam), "--alloc", a,
           "--offset", str(off), "--seeds", str(n), "--out", out]
    with open(out_abs + ".log", "w") as logf, \
         open(out_abs + ".err.log", "w") as errf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf)
    active.append((job, p))
    print(f"start lam={lam} alloc={a} off={off} pid={p.pid}", flush=True)


while idx < len(NEED) or active:
    while idx < len(NEED) and len(active) < MAX_CONCURRENT:
        launch(NEED[idx])
        idx += 1
    done = [a for a in active if a[1].poll() is not None]
    for job, p in done:
        lam, a, off, n, out = job
        print(f"done lam={lam} alloc={a} off={off} rc={p.returncode} "
              f"[{(time.perf_counter()-t0)/60:.1f}m]", flush=True)
        active.remove((job, p))
    if not active:
        break
    time.sleep(5)

print(f"resume finished in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

# Merge all chunks
merged = []
missing = []
for out in ALL_JOBS:
    p = os.path.join(RAW, out)
    if os.path.exists(p):
        with open(p) as f:
            merged += json.load(f)
        print(f"chunk {out}: done", flush=True)
    else:
        missing.append(out)
print(f"missing chunks: {missing}", flush=True)

final = os.path.join(RAW, "lambda_sweep.json")
with open(final, "w") as f:
    json.dump(merged, f, indent=2, default=str)
print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
