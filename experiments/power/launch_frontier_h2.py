#!/usr/bin/env python
"""Launch Frontier/H2 reserve campaign: 80 seeds x Frontier under bands_rich.
8 subprocess workers (8 chunks of 10 seeds), then merge -> p5h_frontier_H2.json.
"""
import subprocess
import sys
import os
import json

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_revision.py")
RAW = os.path.join(ROOT, "results", "raw")

CHUNK_OFFSETS = list(range(10, 90, 10))  # 10,20,...,80 -> 8 chunks of 10
NPER = 10
DET = "bands_rich"
ALLOC = "frontier"

procs = []
for ci, offset in enumerate(CHUNK_OFFSETS):
    out = f"p5h_frontier_H2_{ci}.json"
    out_abs = os.path.join(RAW, out)
    for suf in ("", ".log", ".err.log"):
        p = out_abs + suf
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, SCRIPT, "--offset", str(offset),
           "--seeds", str(NPER), "--cov", "0.95",
           "--detectability", DET, "--band-pd", "0.5,0.7,0.9",
           "--alloc", ALLOC, "--out", out]
    with open(out_abs + ".log", "w") as logf, \
         open(out_abs + ".err.log", "w") as errf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf,
                             creationflags=subprocess.CREATE_NO_WINDOW)
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
final = os.path.join(RAW, "p5h_frontier_H2.json")
with open(final, "w") as f:
    json.dump(merged, f, indent=2)
print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
print("DONE", flush=True)