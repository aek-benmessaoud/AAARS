"""Profile coverage + risk under each allocation to test separation."""
import sys, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 2000, "trace_stride": 50,
       "aaars": {"w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
                 "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                 "blend_threshold": 0.30, "num_bins": 5}}

for alloc in ["boustro", "minerich"]:
    r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    valid = [e for e in trace if e.get("coverage", -1) >= 0]
    
    print(f"\n=== {alloc} ===")
    for idx in [0, len(valid)//4, len(valid)//2, -1]:
        e = valid[idx]
        print(f"  t={e['t']:>5d} | "
              "cov={:.3f} ".format(e.get('coverage',0)) +
              "deficit={:.3f} ".format(e.get('coverage_deficit',0)) +
              "FI={:.3f} ".format(e.get('frequency_instability',0)) +
              "risk={:.3f} ".format(e.get('risk_score',0)) +
              "etype={}".format(e.get('estimator_type','?')))
