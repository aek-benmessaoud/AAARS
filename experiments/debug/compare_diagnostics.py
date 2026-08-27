"""Compare AAARS diagnostics under boustro vs minerich."""
import sys, os, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 300, "aaars": {
    "w_temporal": 0.0, "w_spatial": 0.45, "w_frequency": 0.55,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
    "blend_threshold": 0.30},
    "trace_stride": 30}

for alloc in ["boustro", "minerich"]:
    print(f"\n=== {alloc.upper()} ===")
    r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    for e in trace:
        if e["t"] >= 60:
            print(f"  t={e['t']:3d}  risk={e.get('risk_score',0):.3f}  "
                  f"TC={e.get('temporal_clustering',-1):.3f}  "
                  f"SC={e.get('spatial_concentration',-1):.3f}  "
                  f"FI={e.get('frequency_instability',-1):.3f}  "
                  f"RB={e.get('richness_bias',-1):.3f}  "
                  f"n_det={e['belief_ndet']}  f1={e['f1']:.0f}  f2={e['f2']:.0f}")
