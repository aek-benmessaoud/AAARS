"""Analyze why AAARS risk scores are identical under boustro vs minerich."""
import sys, os, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG
import numpy as np

cfg = {**DEFAULT_CFG, "max_steps": 2000, "aaars": {
    "w_temporal": 0.4, "w_spatial": 0.3, "w_frequency": 0.3,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}}

# Run one episode each, collect traces
for alloc in ["boustro", "minerich"]:
    r = run_episode(alloc, 10, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    
    print(f"\n{'='*50}")
    print(f"{alloc} — risk={r['aaars__final_risk']:.4f}, switches={r['aaars__switches']}")
    print(f"{'='*50}")
    
    # Show diagnostic evolution at key points
    for entry in trace[::50]:  # every 50th entry
        t = entry["t"]
        risk = entry.get("risk_score", 0)
        f1 = entry.get("f1", 0)
        f2 = entry.get("f2", 0)
        f1f2 = entry.get("f1_f2_ratio", 0)
        tc = entry.get("temporal_clustering", 0)
        sc = entry.get("spatial_concentration", 0)
        fi = entry.get("frequency_instability", 0)
        print(f"  t={t:>4d} risk={risk:.3f}  f1={f1:.0f} f2={f2:.0f} f1/f2={f1f2:.1f}  "
              f"TC={tc:.3f} SC={sc:.3f} FI={fi:.3f}")
