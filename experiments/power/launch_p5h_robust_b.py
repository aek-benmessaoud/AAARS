#!/usr/bin/env python
"""Completion wave B for the Item-6 robustness campaign.

Wave A ran only the low half of seeds per cell (40 seeds/cell). This wave runs
the OTHER 40 seeds (offsets 10,30,50,70 per cell -> full 80 seeds/cell) and
merges into the existing p5h_obstacles.json / p5h_commdelay.json.

Cells needing the missing half:
  obs_10 : already 10,30  -> need 50,70
  obs_20 : already 50,70  -> need 10,30
  dly_02 : already 10,30  -> need 50,70
  dly_04 : already 50,70  -> need 10,30
"""
import subprocess, sys, os, json

ROOT = r"F:\Project11-AAARS"
SCRIPT = os.path.join(ROOT, "experiments", "power", "power_revision.py")
RAW = os.path.join(ROOT, "results", "raw")

CELLS = [
    ("obs_10", ["--obstacle-ratio", "0.10"], [50, 70]),
    ("obs_20", ["--obstacle-ratio", "0.20"], [10, 30]),
    ("dly_02", ["--comm-delay", "2"], [50, 70]),
    ("dly_04", ["--comm-delay", "4"], [10, 30]),
]
NPER = 20
LEVER_OF = {"obs_10": "obstacles", "obs_20": "obstacles",
            "dly_02": "commdelay", "dly_04": "commdelay"}

procs = []
for tag, extra, offsets in CELLS:
    for ci, offset in enumerate(offsets):
        out = f"p5h_rob_{tag}_B{chr(ord('A')+ci)}.json"
        out_abs = os.path.join(RAW, out)
        for suf in ("", ".log", ".err.log"):
            p = out_abs + suf
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, SCRIPT, "--offset", str(offset),
               "--seeds", str(NPER), "--cov", "0.95", *extra, "--out", out]
        with open(out_abs + ".log", "w") as logf, \
             open(out_abs + ".err.log", "w") as errf:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=errf,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((tag, offset, out, p))
        print(f"started {tag} offset={offset} out={out} pid={p.pid}", flush=True)

print("waiting for completion...", flush=True)
for tag, offset, out, p in procs:
    rc = p.wait()
    print(f"{tag} offset={offset} out={out} rc={rc}", flush=True)

# Merge new chunks into existing lever files, dedup by (alloc, run).
for lev in ("obstacles", "commdelay"):
    final = os.path.join(RAW, f"p5h_{lev}.json")
    existing = json.load(open(final)) if os.path.exists(final) else []
    seen = {(r["alloc"], r["run"]) for r in existing}
    for tag, offset, out, p in procs:
        if LEVER_OF[tag] != lev:
            continue
        path = os.path.join(RAW, out)
        if os.path.exists(path):
            for r in json.load(open(path)):
                key = (r["alloc"], r["run"])
                if key not in seen:
                    existing.append(r); seen.add(key)
    with open(final, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"MERGED -> {final} ({len(existing)} records)", flush=True)
print("DONE", flush=True)