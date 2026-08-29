"""
percepts/io.py — (de)serialisation of a PerceptStep stream for replay (D5).

A PerceptStep is frozen and carries a uint8 bits grid + scalar belief stats. To
replay a mission offline (without re-simulating), we record the per-step
PerceptStep stream to disk and feed it back through the same stopping rules.

The bits grid is compressed with base64 (its byte layout would otherwise bloat
the trace). Ground truth never enters the record — only the leak-free fields.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np

from src.percepts import PerceptStep


def _pack_bits(bits: np.ndarray) -> str:
    arr = np.ascontiguousarray(bits, dtype=np.uint8)
    return base64.b64encode(zlib.compress(arr.tobytes(), 9)).decode("ascii")


def _unpack_bits(payload: str, grid_size: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    return np.frombuffer(raw, dtype=np.uint8).reshape(grid_size, grid_size)


def percept_to_dict(p: PerceptStep, grid_size: int) -> dict:
    """Serialize a PerceptStep (no ground truth) to a JSON-friendly dict."""
    return {
        "t": int(p.t),
        "bits_b64": _pack_bits(p.bits),
        "fleet_coverage": float(p.fleet_coverage),
        "new_cells_scanned": int(p.new_cells_scanned),
        "new_finds": int(p.new_finds),
        "coverage_frac": float(p.coverage_frac),
    }


def percept_from_dict(d: dict, grid_size: int) -> PerceptStep:
    """Reconstruct a PerceptStep from the dict produced by percept_to_dict."""
    return PerceptStep(
        t=int(d["t"]),
        bits=_unpack_bits(d["bits_b64"], grid_size),
        fleet_coverage=float(d.get("fleet_coverage", 0.0)),
        new_cells_scanned=int(d.get("new_cells_scanned", 0)),
        new_finds=int(d.get("new_finds", 0)),
        coverage_frac=float(d.get("coverage_frac", 0.0)),
    )
