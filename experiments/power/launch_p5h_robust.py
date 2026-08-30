#!/usr/bin/env python
"""Launch the Item-6 robustness campaign (parallel, non-blocking, 8 workers).

Levers (K=60, N=6, 80 seeds each, both allocations boustro+minerich):
  obstacles : obstacle_ratio in {0.10, 0.20}  -> 2 x 160 = 320 eps
  comm-delay: comm_delay in {2, 4}            -> 2 x 160 = 320 eps
Total 640 episodes across 8 detached subprocess workers (4 per lever).

Merges into:
  results/raw/p5h_obstacles.json  (records carry obstacle_ratio)
  results/raw/p5h_commdelay.json  (records carry comm_delay)
"""
import subprocess
import sys
import os
import json

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_revision.py")
RAW = os.path.join(ROOT, "results", "raw")

# Each cell: (tag, [extra args], chunk offsets), 20 seeds per chunk, both allocs.
CELLS = [
    ("obs_10", ["--obstacle-ratio", "0.10"], [10, 30]),
    ("obs_20", ["--obstacle-ratio", "0.20"], [50, 70]),
    ("dly_02", ["--comm-delay", "2"], [10, 30]),
    ("dly_04", ["--comm-delay", "4"], [50, 70]),
]
NPER = 20
COV = 0.95

LEVER_OF = {"obs_10": "obstacles", "obs_20": "obstacles",
            "dly_02": "commdelay", "dly_04": "commdelay"}

procs = []
for tag, extra, offsets in CELLS:
    for ci, offset in enumerate(offsets):
        out = f"p5h_rob_{tag}_{chr(ord('A')+ci)}.json"
        out_abs = os.path.join(RAW, out)
        for suf in ("", ".log", ".err.log"):
            p = out_abs + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, SCRIPT, "--offset", str(offset),
               "--seeds", str(NPER), "--cov", str(COV),
               *extra, "--out", out]
        with open(out_abs + ".log", "w") as logf, \
             open(out_abs + ".err.log", "w") as errf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((tag, offset, out, p))
        print(f"started {tag} chunk {chr(ord('A')+ci)} offset={offset} "
              f"out={out} pid={p.pid}", flush=True)

print("waiting for completion...", flush=True)
for tag, offset, out, p in procs:
    rc = p.wait()
    print(f"{tag} offset={offset} out={out} rc={rc}", flush=True)

merged = {}
for tag, offset, out, p in procs:
    lev = LEVER_OF[tag]
    merged.setdefault(lev, [])
    path = os.path.join(RAW, out)
    if os.path.exists(path):
        with open(path) as f:
            merged[lev] += json.load(f)
for lev, recs in merged.items():
    final = os.path.join(RAW, f"p5h_{lev}.json")
    with open(final, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"MERGED -> {final} ({len(recs)} records)", flush=True)
print("DONE", flush=True)