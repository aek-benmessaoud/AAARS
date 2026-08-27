"""Quick targeted test: does AAARS with coverage-based risk stop correctly?"""
import sys, json
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

cfg = {**DEFAULT_CFG, "max_steps": 6000,
       "aaars": {"w_temporal": 0.0, "w_coverage": 0.55, "w_frequency": 0.45,
                 "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0,
                 "blend_threshold": 0.30, "num_bins": 5}}

for seed in [10, 11, 12, 13, 14]:
    r = run_episode("boustro", seed, seed*1000, cfg=cfg, collect_trace=False)
    t = r.get("aaars__t")
    rec = r.get("aaars__recall", 0)
    risk = r.get("aaars__final_risk", 0)
    print(f"  boustro seed={seed}: aaars_t={t} rec={rec:.1f}% risk={risk:.3f}")

for seed in [10, 11, 12, 13, 14]:
    r = run_episode("minerich", seed, seed*1000, cfg=cfg, collect_trace=False)
    t = r.get("aaars__t")
    rec = r.get("aaars__recall", 0)
    risk = r.get("aaars__final_risk", 0)
    print(f"  minerich seed={seed}: aaars_t={t} rec={rec:.1f}% risk={risk:.3f}")
