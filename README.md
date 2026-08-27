# Project11-AAARS

**Allocation-Aware Adaptive Richness Stopping.**

An online, ground-truth-free adaptive stopping rule for richness certification. AAARS continuously diagnoses the fleet's own observation process (temporal clustering, richness-bias, and explicit *coverage deficit*), maps it to a risk score `r(t)`, and uses that score two ways:

1. **Continuous estimator blending**: `a_hat = (1-r_eff)*Chao1 + r_eff*Chao92` with a dead zone so systematic runs keep pure Chao1.
2. **Adaptive certification**: `alpha_adj = alpha0 / (1 + lambda*r)`, tightening the threshold under risk.

## Layout

```
src/aaars/            diagnostics + controller
src/estimators/       Chao1, Chao92, blended bounds
src/environment/      grid + mine environment (leak-free)
src/allocation/       BoustroLanes, MineRichness
src/experiments/      unified runner
experiments/          confirmation, sweep, ablation, debug
results/figures/      generated figures
manuscript.md         the paper
```

## Reproduce

```powershell
# tests
python -m pytest tests -q

# confirmation (20 seeds x 2 allocs)
python experiments/confirmation/confirm_base.py

# K x N sweep (90 episodes)
python experiments/sweep/config_sweep.py

# figures + stats
python results/figures/generate_figures.py
python results/paper_stats.py
```

## Headline result

Under a biased MineRichness allocation, Chao1-CI false-certifies (recall < 95%) in **25%** of 20 episodes; AAARS reduces this to **15%** while never false-certifying under the systematic BoustroLanes policy and preserving the stopping rate. Across K×N, AAARS never hurts and gives up to **40-pp** reductions at high agent counts.

Python: `C:\Python314\python.exe`
