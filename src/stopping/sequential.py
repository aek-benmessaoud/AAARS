"""
stopping/sequential.py — Modern anytime-valid stopping baselines for AAARS.

Two statistically-motivated sequential stopping rules that are INDEPENDENT of
the Chao richness estimators, added as "modern baseline" comparators.

Both are LEAK-FREE: they consume only per-step detection / survey counts
(derived from the fleet scan/bits streams) and never ground truth.

THE SCIENTIFIC POINT
--------------------
Both rules read "new finds per unit of ADDITIONAL survey effort" from the same
leak-free detection / coverage stream, but encode opposite assumptions about
how to declare exhaustion:

  - GapSPRT (naive rate rule) stops as soon as the stream of new finds goes
    silent for long enough. Under biased allocation (MineRichness / hotspot /
    greedy re-scan the same rich cells) the discovery stream falls quiet while
    most mines are still hidden, so it FALSE-CERTIFIES exhaustion early -- the
    core claim the paper demonstrates. Under a systematic (Boustro) sweep it
    only goes quiet at true exhaustion, so it is (roughly) valid there.

  - RateCS (coverage-aware rule) refuses to certify until the fleet has covered
    a large fraction of the domain AND then a consecutive silent run of new
    cells is observed. It is the safe counterpoint: it does not certify under
    bias (coverage there climbs slowly), firing only near genuine exhaustion.

The contrast between the two -- an unsafe naive rate rule vs a safe
coverage-bearing rule -- on the same biased streams is precisely the finding.

The two constructions:

RateCS   : Silent-run, coverage-gated exhaustion rule. Per step (leak-free): new_cells (distinct cells newly scanned),
           new_find (new confirmed cells), coverage_frac (fraction of the
           domain scanned at least once). It counts a *consecutive* run of
           newly-scanned cells that yield zero new finds (the run resets on
           any find or any step with no new cells), and stops when that silent
           run is long enough AT a coverage level above a threshold.


GapSPRT  : Wald (1945) sequential probability ratio test on the GEOMETRIC
           inter-discovery gap. Let G = # consecutive steps between new finds,
           G ~ Geometric(p) (p = per-step prob of finding a new mine; takes
           values where a find occurs). H0: p >= p0 (active discovery, keep
           searching) vs H1: p <= p1 (exhausted) with p1 < p0. A well-specified
           likelihood avoids the naive per-step-Poisson SPRT that fires on the
           opening burst. "Exhausted" is accepted only after observing
           genuinely long gaps -- which biased allocation produces spuriously.
"""

import numpy as np
from src.percepts import PerceptStep


class RateCS:
    """Coverage-aware silence rule for survey exhaustion.

    Observable per step (leak-free):
        new_cells : distinct cells newly scanned this step (>=0)
        new_find  : new confirmed cells this step (>=0)
        coverage_frac : fraction of the plausible domain whose cells have been
                        scanned at least once

    Rationale: "exhaustion" of a surveyed area is signalled by a *silent run*
    -- sweeping a substantial number of previously-unseen cells turns up no
    new confirmed cells -- while the fleet has already covered a large fraction
    of the domain.  This is conservative by construction: it refuses to
    certify when only a small part of the area has been surveyed (the biased
    re-scanning regime), so it should trigger late (near-truth) under a
    systematic sweep and stay safe under allocation bias -- the designed
    contrast to the naive per-step rate rule (GapSPRT), which spuriously
    certifies early under bias.
    """

    def __init__(self, min_silent_cells=8, min_coverage=0.90):
        self.min_silent_cells = min_silent_cells
        self.min_coverage = min_coverage
        self.reset()

    def reset(self):
        self.stopped = False
        self._cells = 0.0      # consecutive newly-scanned cells with no find

    def update(self, percept: "PerceptStep"):
        if self.stopped:
            return True
        new_cells = percept.new_cells_scanned
        new_find = percept.new_finds
        coverage_frac = percept.coverage_frac
        if new_cells > 0 and new_find <= 0:
            self._cells += float(new_cells)
        else:
            # a find, or a step with no new cells: break the silent run
            self._cells = 0.0
        if coverage_frac >= self.min_coverage and \
           self._cells >= self.min_silent_cells:
            self.stopped = True
            return True
        return False


class GapSPRT:
    """Wald (1945) sequential probability ratio test on the GEOMETRIC
    inter-discovery gap.

    Model: each scan-step independently yields a new confirmed cell with
    probability p.  The run of empty steps between finds is geometric.  We test
        H0 : p >= p0   (still actively discovering -> keep searching)
        H1 : p <= p1   (effectively exhausted -> stop),  p1 < p0.
    The log-likelihood-ratio accumulates step by step: +L(p1)/L(p0) for each
    empty step (survival piece, favors H1 as gaps grow) and +log(p1/p0) for
    each find (favors H0).  Per Wald, accept H1 (stop) when the accumulated
    logLR crosses the UPPER boundary  B = log((1-beta)/alpha).

    This is the honest rate-rule: at the naive per-step level it cannot tell
    "genuinely exhausted" from "biased re-scanning returns no new finds", so it
    stops early whenever the discovery stream goes quiet -- which is exactly the
    false-certification claim under allocation bias.
    """

    def __init__(self, p0=0.6, p1=0.03, alpha=0.05, beta=0.05,
                 min_silent_steps=4):
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = alpha
        self.beta = beta
        self.min_silent_steps = min_silent_steps
        self.reset()

    @property
    def _surv(self):
        return np.log(1.0 - self.p1) - np.log(1.0 - self.p0)

    @property
    def _upper(self):
        return np.log((1.0 - self.beta) / self.alpha)

    def reset(self):
        self.stopped = False
        self.logLR = 0.0
        self.silent = 0          # consecutive empty steps
        self.finds = 0

    def update(self, percept: "PerceptStep"):
        if self.stopped:
            return True
        new_find = percept.new_finds
        if new_find > 0:
            self.logLR += np.log(self.p1 / self.p0)
            self.silent = 0
            self.finds += 1
        else:
            self.logLR += self._surv
            self.silent += 1
        # Accept H1 (exhausted) on the upper boundary.
        if self.silent >= self.min_silent_steps and \
           self.logLR >= self._upper:
            self.stopped = True
            return True
        return False
