#!/usr/bin/env python
"""Launch power_baselines.py across 4 parallel chunks, then merge outputs."""
import subprocess
import sys
import os
import json

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_baselines.py")
RAW = os.path.join(ROOT, "results", "raw")

CHUNKS = [
    (10, "baselines_A.json"),
    (30, "baselines_B.json"),
    (50, "baselines_C.json"),
    (70, "baselines_D.json"),
]
NPER = 20
COV = 0.95

procs = []
for offset, out in CHUNKS:
    out_abs = os.path.join(RAW, out)
    for suf in ("", ".log", ".err.log"):
        p = out_abs + suf
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, SCRIPT, "--offset", str(offset), "--seeds", str(NPER),
           "--cov", str(COV), "--out", out]
    with open(out_abs + ".log", "w") as logf, \
         open(out_abs + ".err.log", "w") as errf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf)
    procs.append((offset, out, p))
    print(f"started offset={offset} out={out} pid={p.pid}", flush=True)

print("waiting for completion...", flush=True)
for offset, out, p in procs:
    rc = p.wait()
    print(f"offset={offset} out={out} rc={rc}", flush=True)

merged = []
for offset, out, p in procs:
    path = os.path.join(RAW, out)
    if os.path.exists(path):
        with open(path) as f:
            merged += json.load(f)
final = os.path.join(RAW, "power_baselines.json")
with open(final, "w") as f:
    json.dump(merged, f, indent=2)
print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
