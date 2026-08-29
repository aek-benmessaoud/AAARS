"""
percept_step.py — PerceptStep: the single leak-free observation type.

A PerceptStep is the ONLY input a stopping rule may consume. It is a plain
immutable dataclass whose fields are restricted to fleet-observable belief
statistics (scan/fusion bits, survey yield, fleet coverage). Ground-truth
quantities (true mine count K, located/undetected masks, true recall) have no
representation here — it is structurally impossible for a rule that takes a
PerceptStep to reach env.n_detections or any ground-truth channel.

Why this exists (Deliverable 3 of the leak-free type refactor):
  * Before, the runner threaded raw env state (`bits`, `fleet_coverage`,
    `new_cells`, ...) into each rule as separate positional arguments. Nothing
    prevented a future edit from passing `env.n_detections` into a rule.
  * After, every rule signature is `update(percept: PerceptStep)`, and the
    runner constructs the PerceptStep in exactly one place from belief data.
    Leak-freedom is enforced by construction, not by audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PerceptStep:
    """One leak-free observation step handed to every stopping rule."""

    # Causal step index (the mission clock; not ground truth)
    t: int

    # Fleet scan/fusion bits (uint8 grid): bit i of a cell set if mine hit in
    # occasion i. This is the shared per-cell belief representation.
    bits: np.ndarray = field(repr=False)

    # Fraction of traversable cells scanned at least once by the fleet. This
    # is the fleet's own measured sweep, always leak-free.
    fleet_coverage: float = 0.0

    # Survey yield this step (from the same bit stream — leak-free).
    new_cells_scanned: int = 0        # distinct cells newly scanned
    new_finds: int = 0                # new confirmed cells this step
    coverage_frac: float = 0.0        # fraction of plausible domain scanned

    def require_bits(self) -> np.ndarray:
        """Return the bits grid (rules use this instead of raw env state)."""
        return self.bits

    def as_dict(self) -> dict:
        """Serialisable summary for trace output (no ground truth)."""
        return {
            "t": self.t,
            "fleet_coverage": round(self.fleet_coverage, 6),
            "new_cells_scanned": int(self.new_cells_scanned),
            "new_finds": int(self.new_finds),
            "coverage_frac": round(self.coverage_frac, 6),
        }
