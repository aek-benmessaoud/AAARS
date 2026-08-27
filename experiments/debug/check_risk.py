"""Quick diagnostic: what risk scores does AAARS produce under each allocation?"""
import sys, json, random
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 3000, "trace_stride": 20,
       "aaars": {"w_temporal": 0.0, "w_spatial": 0.45, "w_frequency": 0.55,
                 "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                 "blend_threshold": 0.30}}

for alloc in ["boustro", "minerich"]:
    r = run_episode(alloc, 42, 10000, cfg=cfg, collect_trace=True)
    trace = json.loads(r["_trace"])
    valid = [e for e in trace if e.get("temporal_clustering", -1) >= 0]
    risk = [e["risk_score"] for e in valid]
    tc = [e["temporal_clustering"] for e in valid]
    sc = [e["spatial_concentration"] for e in valid]
    fi = [e["frequency_instability"] for e in valid]
    print(f"{alloc}:")
    print(f"  risk: min={min(risk):.3f} max={max(risk):.3f} median={risk[len(risk)//2]:.3f}")
    print(f"  tc:   min={min(tc):.3f} max={max(tc):.3f} median={tc[len(tc)//2]:.3f}")
    print(f"  sc:   min={min(sc):.3f} max={max(sc):.3f} median={sc[len(sc)//2]:.3f}")
    print(f"  fi:   min={min(fi):.3f} max={max(fi):.3f} median={fi[len(fi)//2]:.3f}")
    print(f"  last5: risk={[f'{r:.3f}' for r in risk[-5:]]} tc={[f'{x:.3f}' for x in tc[-5:]]}")
    print()
