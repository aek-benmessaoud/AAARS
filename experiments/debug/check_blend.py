"""Quick check: does the blend activate under each allocation?"""
import sys, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 1500, "trace_stride": 50,
       "aaars": {"w_temporal": 0.0, "w_spatial": 0.45, "w_frequency": 0.55,
                 "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                 "blend_threshold": 0.10, "num_bins": 5}}

for alloc in ["boustro", "minerich"]:
    r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    valid = [e for e in trace if e.get("temporal_clustering", -1) >= 0]
    
    print(f"\n=== {alloc} ===")
    blend_active = 0
    for e in valid:
        risk = e.get("risk_score", 0)
        etype = e.get("estimator_type", "?")
        if etype in ("blended", "chao92"):
            blend_active += 1
        if e["t"] % 200 == 0 or e["t"] == valid[-1]["t"]:
            print(f"  t={e['t']:>5d} risk={risk:.3f} type={etype:8s} "
                  f"SC={e.get('spatial_concentration',0):.3f} "
                  f"FI={e.get('frequency_instability',0):.3f} "
                  f"alpha={e.get('aaars_alpha_adj',0):.5f}")
    
    print(f"  Blend active steps: {blend_active}/{len(valid)}")
    print(f"  Estimator type counts: ", end="")
    from collections import Counter
    types = Counter(e.get("estimator_type", "?") for e in valid)
    print(dict(types))
