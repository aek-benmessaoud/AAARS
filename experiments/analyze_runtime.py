#!/usr/bin/env python
"""Summarize runtime_bench.py per-step microbench results."""
import os
import glob
import json
import numpy as np

RAW = r"F:\Project11-AAARS\results\raw"
METHODS = ["chao1", "aaars", "gap_sprt"]

rows = []
for f in sorted(glob.glob(os.path.join(RAW, "bench_*.json"))):
    r = json.load(open(f))
    us = {m: 1e6 * r["total_s"][m] / r["n_steps"][m] for m in METHODS}
    rows.append((r["alloc"], r["seed"], us))

print("per-step microseconds (controller/estimator update only):")
print("  alloc     seed   chao1   aaars  gap_sprt   aaars-chao1   ratio")
for alloc, seed, us in rows:
    print(f"  {alloc:8s} {seed:4d} {us['chao1']:7.1f} {us['aaars']:7.1f} "
          f"{us['gap_sprt']:7.2f}   {us['aaars']-us['chao1']:+7.1f}  "
          f"{us['aaars']/us['chao1']:5.2f}x")

ca = [r[2]["chao1"] for r in rows]
aa = [r[2]["aaars"] for r in rows]
print()
print(f"MEAN per-step: chao1={np.mean(ca):.1f} us, AAARS={np.mean(aa):.1f} us, "
      f"added={np.mean(aa)-np.mean(ca):+.1f} us, ratio={np.mean(aa)/np.mean(ca):.2f}x")
print("Note: this is the pure estimator/controller update. Full env+perception"
      " dominates the wall time; the add-on is what AAARS costs beyond Chao1.")
