"""Quantum error-correction simulator.

A small NumPy state-vector engine plus the 3-qubit bit-flip, 3-qubit
phase-flip and 9-qubit Shor codes, a Monte Carlo logical-error-rate
estimator, an automated noise-configuration sweep, analytic cross-checks and
Matplotlib reporting.
"""

from __future__ import annotations

from .codes import CODES, BitFlipCode, PhaseFlipCode, ShorCode, Unencoded
from .montecarlo import simulate
from .noise import CHANNELS

__version__ = "0.1.0"

__all__ = [
    "CODES",
    "CHANNELS",
    "BitFlipCode",
    "PhaseFlipCode",
    "ShorCode",
    "Unencoded",
    "simulate",
    "__version__",
]
