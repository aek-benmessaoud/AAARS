#!/usr/bin/env python
"""
replay_episode.py — deliverable D5: replay a saved PerceptStep stream through
the stopping rules WITHOUT re-simulating the environment.

This cashes out the paper's "honest, reproducible evaluation" claim: a reviewer
(or reader) can take a recorded per-step PerceptStep stream and re-derive every
leak-free stopping decision offline, independent of the environment simulator.

Record a stream with:
    result = run_episode(alloc, seed, env_seed,
                         cfg=cfg, collect_percepts=True)
    json.dump(json.loads(result["_percepts"]), open(out, "w"))

Then replay:
    python experiments/replay/replay_episode.py <stream.json> [--seed 0 ...]

Rules reproduced (all pure functions of the PerceptStep stream):
    chao1_ci, chao92_ci, aaars, discrete_aaars, threshold_aaars,
    coverage_only, rate_cs, gap_sprt, diminishing.

NOT reproduced here (not leak-free-replayable from PerceptStep):
    - fixed_2 : needs the full per-cell scan-count mask (env.fleet_scan_max).
    - oracle_95 : ground-truth evaluation bound, by definition.
Run with --compare to a fresh live episode to confirm bit-for-bit agreement on
the reproduced rules.
"""

import json
import os
import sys
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.percepts import PerceptStep
from src.percepts.io import percept_from_dict
from src.estimators.chao1 import residual_estimate
from src.estimators.chao92 import chao92_from_freq, full_frequencies
from src.aaars.controller import AAARSController
from src.aaars.discrete_selector import DiscreteSelectorController
from src.stopping.diminishing import DiminishingStop
from src.stopping.sequential import RateCS, GapSPRT
from src.experiments.runner import DEFAULT_CFG

REPRODUCED_RULES = [
    "chao1_ci", "chao92_ci", "aaars", "discrete_aaars",
    "threshold_aaars", "coverage_only", "rate_cs", "gap_sprt",
    "diminishing",
]


def build_rules(cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    ak = cfg.get("aaars", {})
    aaars = AAARSController(
        window_size=ak.get("window_size", 8), grid_size=cfg["grid_size"],
        num_bins=ak.get("num_bins", 5), w_temporal=ak.get("w_temporal", 0.0),
        w_coverage=ak.get("w_coverage", 0.55), w_frequency=ak.get("w_frequency", 0.45),
        ema_alpha=ak.get("ema_alpha", 0.1), base_alpha=ak.get("base_alpha", 0.05),
        risk_lambda=ak.get("risk_lambda", 1.0),
        blend_threshold=ak.get("blend_threshold", 0.30))
    discrete = DiscreteSelectorController(
        window_size=ak.get("window_size", 8), grid_size=cfg["grid_size"])
    diminishing = DiminishingStop(tau=150, n_min=5)
    rate_cs = RateCS(min_silent_cells=8, min_coverage=0.90)
    gap_sprt = GapSPRT(p0=0.6, p1=0.03, alpha=0.05, beta=0.05)
    return aaars, discrete, diminishing, rate_cs, gap_sprt


def replay(stream, cfg=None):
    """Replay a PerceptStep stream; return dict of {rule: stop_t}."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    aaars, discrete, diminishing, rate_cs, gap_sprt = build_rules(cfg)
    stop_t = {r: None for r in REPRODUCED_RULES}

    for i, dd in enumerate(stream):
        t = dd["t"]
        percept = percept_from_dict(dd, cfg["grid_size"])
        bits = percept.bits

        est1 = residual_estimate(bits)
        f2_floor = max(5.0, np.ceil(0.08 * est1["n_det"]))
        armed1 = est1["n_det"] >= 5 and est1["f2"] >= f2_floor

        fk = full_frequencies(bits)
        est92 = chao92_from_freq(fk)
        f2_floor92 = max(5.0, np.ceil(0.08 * est92["n_det"]))
        armed92 = est92["n_det"] >= 5 and float(fk[2]) >= f2_floor92

        if stop_t["chao1_ci"] is None and armed1:
            if est1["ci_upper"] <= 0.05 * est1["K_hat"]:
                stop_t["chao1_ci"] = t
        if stop_t["chao92_ci"] is None and armed92:
            if est92["U92"] >= 0 and est92["ci92_upper"] <= 0.05 * est92["K_hat92"]:
                stop_t["chao92_ci"] = t
        if stop_t["rate_cs"] is None and rate_cs.update(percept):
            stop_t["rate_cs"] = t
        if stop_t["gap_sprt"] is None and gap_sprt.update(percept):
            stop_t["gap_sprt"] = t
        if stop_t["diminishing"] is None and diminishing.update(t, est1["n_det"]):
            stop_t["diminishing"] = t

        aaars_result = aaars.step(percept)
        if stop_t["aaars"] is None and aaars_result["stop"]:
            stop_t["aaars"] = t
        alpha_adj = aaars_result.get("alpha_adj", 0.05)
        if stop_t["threshold_aaars"] is None and armed1:
            if est1["ci_upper"] <= alpha_adj * est1["K_hat"]:
                stop_t["threshold_aaars"] = t
        cov_thr = cfg.get("coverage_only_threshold", 0.90)
        if stop_t["coverage_only"] is None and percept.fleet_coverage >= cov_thr:
            stop_t["coverage_only"] = t
        if stop_t["discrete_aaars"] is None and discrete.step(percept)["stop"]:
            stop_t["discrete_aaars"] = t

    return stop_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", help="path to saved _percepts JSON (list)")
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--compare", action="store_true",
                    help="also run a live episode and compare stop times")
    ap.add_argument("--alloc", default="minerich")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override live horizon for --compare (default: len(stream))")
    args = ap.parse_args()

    with open(args.stream, encoding="utf-8") as f:
        stream = json.load(f)
    cfg = {**DEFAULT_CFG, "grid_size": args.grid,
           "max_steps": args.max_steps or len(stream),
           "coverage_only_threshold": 0.95}
    stop_t = replay(stream, cfg=cfg)

    print(f"Replayed {len(stream)} steps from {args.stream}")
    for r in REPRODUCED_RULES:
        print(f"  {r:16s}: t={str(stop_t[r]):>6s}")

    if args.compare:
        from src.experiments.runner import run_episode
        env_seed = args.seed * 1000
        live = run_episode(args.alloc, args.seed, env_seed, cfg=cfg,
                           collect_percepts=True, collect_trace=False)
        matches = all(live.get(f"{r}__t") == stop_t[r] for r in REPRODUCED_RULES)
        print("\n[compare] vs live episode:")
        for r in REPRODUCED_RULES:
            lv = live.get(f"{r}__t")
            ok = "OK " if lv == stop_t[r] else "DIFF"
            print(f"  [{ok}] {r:16s}: live={str(lv):>6s} replay={str(stop_t[r]):>6s}")
        print("\nRESULT:", "REPLAY REPRODUCES LIVE" if matches else "MISMATCH")
        sys.exit(0 if matches else 1)


if __name__ == "__main__":
    main()
