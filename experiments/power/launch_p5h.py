#!/usr/bin/env python
"""Launch the P5-H generalization campaign: 2 regimes (bands_hetero=H1,
bands_rich=H2) x 2 allocations (boustro, minerich) x 80 seeds = 320 episodes.

Mirrors run_revision_parallel.py: 8 detached subprocess workers (4 chunks of
20 seeds per regime, each chunk runs BOTH allocations -> 40 episodes/chunk),
then merges the per-chunk outputs into one file per regime.
"""
import subprocess
import sys
import os
import json
import argparse

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_revision.py")
RAW = os.path.join(ROOT, "results", "raw")

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seeds-per-chunk", type=int, default=20)
parser.add_argument("--detectability", type=str, default=None,
                    help="limit to one regime (bands_hetero|bands_rich)")
parser.add_argument("--tag", type=str, default="p5h")
args = parser.parse_args()

DET_MAP = {"bands_hetero": "H1", "bands_rich": "H2"}
REGIMES = [args.detectability] if args.detectability else list(DET_MAP)

# chunk offsets: 10, 30, 50, 70 (20 seeds each, stride 1000)
CHUNK_OFFSETS = [10, 30, 50, 70]
NPER = args.seeds_per_chunk
COV = 0.95

procs = []
for det in REGIMES:
    label = DET_MAP[det]
    for ci, offset in enumerate(CHUNK_OFFSETS):
        out = f"p5h_{label}_{chr(ord('A')+ci)}.json"
        out_abs = os.path.join(RAW, out)
        for suf in ("", ".log", ".err.log"):
            p = out_abs + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, SCRIPT, "--offset", str(offset),
               "--seeds", str(NPER), "--cov", str(COV),
               "--detectability", det, "--band-pd", "0.5,0.7,0.9",
               "--out", out]
        with open(out_abs + ".log", "w") as logf, \
             open(out_abs + ".err.log", "w") as errf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((det, offset, out, p))
        print(f"started {label} chunk {chr(ord('A')+ci)} "
              f"offset={offset} out={out} pid={p.pid}", flush=True)

print("waiting for completion...", flush=True)
for det, offset, out, p in procs:
    rc = p.wait()
    print(f"{DET_MAP[det]} offset={offset} out={out} rc={rc}", flush=True)

# Merge per regime
for det in REGIMES:
    label = DET_MAP[det]
    merged = []
    for det2, offset, out, p in procs:
        if det2 != det:
            continue
        path = os.path.join(RAW, out)
        if os.path.exists(path):
            with open(path) as f:
                recs = json.load(f)
                merged += recs
    final = os.path.join(RAW, f"p5h_{label}.json")
    with open(final, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"MERGED -> {final} ({len(merged)} records)", flush=True)
print("DONE", flush=True)