"""Profile actual diagnostic values under each allocation to understand scaling."""
import sys, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 1500, "trace_stride": 50,
       "aaars": {"w_temporal": 0.0, "w_spatial": 0.45, "w_frequency": 0.55,
                 "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                 "blend_threshold": 0.30}}

for alloc in ["boustro", "minerich"]:
    r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    valid = [e for e in trace if e.get("temporal_clustering", -1) >= 0]
    
    print(f"\n=== {alloc} ===")
    # Sample at different stages
    for idx in [0, len(valid)//4, len(valid)//2, -1]:
        e = valid[idx]
        print(f"  t={e['t']:>5d} | n_det={e['belief_ndet']:>4d} "
              f"TC={e.get('temporal_clustering',0):.3f} "
              "SC={:.3f} ".format(e.get('spatial_concentration',0)) +
              "FI={:.3f} ".format(e.get('frequency_instability',0)) +
              "RB={:.3f} ".format(e.get('richness_bias',0)) +
              "risk={:.3f} ".format(e.get('risk_score',0)) +
              "f1={:.0f} f2={:.0f}".format(e.get('f1',0), e.get('f2',0)))
