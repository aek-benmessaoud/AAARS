import sys, os, time
sys.path.insert(0, r"F:\Project11-AAARS")
from src.experiments.runner import run_episode, DEFAULT_CFG

# Test: 100x100, K=60, N=6, 200 steps (just to measure per-step cost)
cfg = {**DEFAULT_CFG, "max_steps": 200, "aaars": {
    "w_temporal": 0.4, "w_spatial": 0.3, "w_frequency": 0.3,
    "ema_alpha": 0.1, "base_alpha": 0.05, "risk_lambda": 1.0}}

t0 = time.perf_counter()
r = run_episode("boustro", 10, 10000, cfg=cfg, collect_trace=False)
elapsed = time.perf_counter() - t0
print("boustro 200 steps: %.1f s (%.3f s/step)" % (elapsed, elapsed/200))

t0 = time.perf_counter()
r = run_episode("minerich", 10, 10000, cfg=cfg, collect_trace=False)
elapsed = time.perf_counter() - t0
print("minerich 200 steps: %.1f s (%.3f s/step)" % (elapsed, elapsed/200))

# Estimate for 6000 steps
boustro_per_step = elapsed / 200  # rough
print("Estimated boustro 6000 steps: %.0f min" % (boustro_per_step * 6000 / 60))
print("Estimated 40 episodes: %.0f hours" % (boustro_per_step * 6000 * 40 / 3600))
