"""
percepts — the leak-free observation boundary for stopping rules.

A single shared type, PerceptStep, is the only input any controller/estimator/
stopping rule may consume. Ground truth has no representation here.
"""

from src.percepts.percept_step import PerceptStep

__all__ = ["PerceptStep"]
