# Project11-AAARS

**Allocation-Aware Adaptive Richness Stopping.**

An online, ground-truth-free adaptive stopping rule for richness certification. AAARS continuously diagnoses the fleet's own observation process (temporal clustering, richness-bias, and explicit *coverage deficit*), maps it to a risk score `r(t)`, and uses the score two ways:

1. **Continuous estimator blending**: `a_hat = (1-r_eff)*Chao1 + r_eff*Chao92` with a dead zone so systematic runs keep pure Chao1.
2. **Adaptive certification**: `alpha_adj = alpha0 / (1 + lambda*r)`, tightening the threshold under risk.

## Layout

```
src/aaars/            diagnostics + controller
src/estimators/       Chao1, Chao92, blended bounds
src/stopping/         stopping rules (chao1_ci, chao92_ci, aaars, ...)
src/environment/      grid + mine environment (leak-free; obstacles, comm-delay)
src/allocation/       BoustroLanes, MineRichness, realistic policies
src/percepts/         leak-free PerceptStep observation stream
src/experiments/      unified runner
experiments/          confirmation, sweep, ablation, power, replay
results/raw/          committed campaign data (.json)
results/figures/      generated figures
tests/                pytest suite
```

## Reproduce

```powershell
# tests
python -m pytest tests -q

# confirmation (80 seeds x 2 allocs) -> results/raw/power_revision.json
python experiments/power/power_revision.py --offset 10 --seeds 80 --alloc all --out power_revision.json

# K x N sweep (90 episodes)
python experiments/sweep/config_sweep.py

# realistic non-adversarial policies
python experiments/power/run_policies_parallel.py

# lambda sensitivity sweep
python experiments/power/lambda_sweep.py

# figures + stats
python results/figures/generate_figures.py
python results/paper_stats.py
```

## Extensions

- **Heterogeneous detectability** (`--detectability bands_hetero|bands_rich`): 3 vertical strips with per-strip `p_d`; `bands_rich` concentrates mines where detection is weakest while holding total `K` invariant.
- **Obstacles** (`--obstacle-ratio`): blocked-cell mask, mines never placed on blocked cells, `K` invariant.
- **Communication delay** (`--comm-delay`): per-agent heterogeneous staleness on rendezvous fusion (async, leak-free by construction).
  Run these via `experiments/power/power_revision.py` (flags: `--detectability`, `--band-pd`, `--obstacle-ratio`, `--comm-delay`); campaigns + analysis under `experiments/power/` (`launch_p5h.py`, `launch_p5h_robust.py`, `analyze_p5h.py`, `analyze_p5h_robust.py`).

## Verification

- `experiments/power/regression_guard.py` recomputes every reported aggregate from the committed `results/raw/*.json` and asserts identity (`ALL MANUSCRIPT FIGURES REPRODUCED`). Run before any re-run / release.
- `experiments/replay/` replays saved PerceptStep traces bit-for-bit through every stopping rule without re-simulating (leak-free by construction).
- `tests/test_no_ground_truth_leakage.py` asserts rules consume only the leak-free observation stream.

Python: `C:\Python314\python.exe`
