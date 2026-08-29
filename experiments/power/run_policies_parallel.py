#!/usr/bin/env python
"""Run realistic-policy confirmation across 6 parallel workers (2 per policy)
and merge results. Policies: frontier, greedy, hotspot; 80 seeds each policy.
"""
import subprocess
import sys
import os
import json

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_revision.py")
RAW = os.path.join(ROOT, "results", "raw")

# 6 workers: 2 per policy, each runs 40 seeds
POLICIES = ["frontier", "greedy", "hotspot"]
SEED_SPLITS = [(10, 40), (50, 40)]  # (offset, num_seeds) per worker per policy
COV = 0.95

procs = []
for pol in POLICIES:
    for (off, n) in SEED_SPLITS:
        out = f"pol_{pol}_{off}.json"
        out_abs = os.path.join(RAW, out)
        for suf in ("", ".log", ".err.log"):
            p = out_abs + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, SCRIPT, "--offset", str(off), "--seeds", str(n),
               "--cov", str(COV), "--out", out, "--alloc", pol]
        with open(out_abs + ".log", "w") as logf, \
             open(out_abs + ".err.log", "w") as errf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf)
        procs.append((pol, off, out, p))
        print(f"started policy={pol} offset={off} out={out} pid={p.pid}",
              flush=True)

print("waiting for all workers...", flush=True)
for pol, off, out, p in procs:
    rc = p.wait()
    print(f"policy={pol} offset={off} out={out} rc={rc}", flush=True)

# Merge all chunk files
merged = []
for pol, off, out, p in procs:
    path = os.path.join(RAW, out)
    if os.path.exists(path):
        with open(path) as f:
            chunk = json.load(f)
        merged += chunk
        print(f"chunk {out}: {len(chunk)} records", flush=True)
    else:
        print(f"WARNING missing {out}", flush=True)

final = os.path.join(RAW, "policies_results.json")
with open(final, "w") as f:
    json.dump(merged, f, indent=2)
print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
