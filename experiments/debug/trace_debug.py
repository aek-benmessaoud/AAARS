"""Check what the AAARS controller actually returns."""
import sys, os, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 200, "aaars": {
    "w_temporal": 0.4, "w_spatial": 0.3, "w_frequency": 0.3,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}}

r = run_episode("boustro", 10, 10000, cfg=cfg, collect_trace=True)
trace = json.loads(r["_trace"])
print("Trace keys:", list(trace[-1].keys()))
print("Trace entry:", trace[-1])
print("\nAll entries:")
for e in trace:
    print("  t=%d risk=%.3f TC=%.3f SC=%.3f FI=%.3f armed=%s" % (
        e["t"], e.get("risk_score", 0),
        e.get("temporal_clustering", -1),
        e.get("spatial_concentration", -1),
        e.get("frequency_instability", -1),
        "yes" if e.get("risk_score", 0) > 0 else "no"))
