"""Closed-form logical error rates used to validate the simulator.

For the two 3-qubit codes under the channel they are designed for (and under
depolarizing noise, where the relevant single-qubit failure probability is
``2p/3``) these expressions are *exact* for the readout metric used in
:mod:`qecsim.codes`.  For the Shor code only a leading-order / upper-bound
estimate is given -- it is a distance-3 code, so its logical error rate scales
as ``O(p^2)`` and is bounded by the probability of two or more physical
faults.
"""

from __future__ import annotations

import numpy as np


def _rep3(q):
    """Logical error of a distance-3 repetition code given per-qubit
    correctable-error probability ``q`` -- i.e. P(2 or 3 of 3 qubits fail)."""
    q = np.asarray(q, dtype=float)
    return 3.0 * q**2 - 2.0 * q**3


def _two_or_more(p, n):
    """P(at least two errors among ``n`` iid qubits)."""
    p = np.asarray(p, dtype=float)
    return 1.0 - (1.0 - p) ** n - n * p * (1.0 - p) ** (n - 1)


def unencoded_logical_error(p, channel, basis="z"):
    """Logical error rate of a bare qubit for the given readout basis."""
    p = np.asarray(p, dtype=float)
    if channel == "depolarizing":
        # Two of the three Paulis flip the readout basis.
        return np.full_like(p, 1.0) * (2.0 * p / 3.0)
    if channel == "bit_flip":
        base = p if basis in ("z", "random") else np.zeros_like(p)
        return base * (0.5 if basis == "random" else 1.0)
    if channel == "phase_flip":
        base = p if basis in ("x", "random") else np.zeros_like(p)
        return base * (0.5 if basis == "random" else 1.0)
    raise ValueError(f"unknown channel {channel!r}")


def code_logical_error(p, code, channel, basis=None):
    """Analytic logical error rate for ``code`` under ``channel``.

    Returns ``np.nan`` (broadcast to ``p``) when no closed form is provided
    for the combination.
    """
    p = np.asarray(p, dtype=float)
    if code == "unencoded":
        return unencoded_logical_error(p, channel, basis or "z")

    if code == "bitflip_3q":
        if channel == "bit_flip":
            return _rep3(p)
        if channel == "depolarizing":
            return _rep3(2.0 * p / 3.0)
        if channel == "phase_flip":
            return np.zeros_like(p)  # Z errors do not disturb a Z-basis readout

    if code == "phaseflip_3q":
        if channel == "phase_flip":
            return _rep3(p)
        if channel == "depolarizing":
            return _rep3(2.0 * p / 3.0)
        if channel == "bit_flip":
            return np.zeros_like(p)  # X errors do not disturb an X-basis readout

    if code == "shor_9q":
        if channel == "depolarizing":
            return _two_or_more(p, 9)  # upper bound; true curve sits below
        if channel in ("bit_flip", "phase_flip"):
            # Only one error type present; a block fails at >=2 faults, any of
            # three blocks, and only the matching half of the random readout
            # basis is sensitive to it.
            return 0.5 * (1.0 - (1.0 - _rep3(p)) ** 3)

    return np.full_like(p, np.nan)


def improvement_factor(p, code, channel, basis=None):
    """Analytic ``unencoded / encoded`` logical-error-rate ratio."""
    enc = code_logical_error(p, code, channel, basis)
    bare = unencoded_logical_error(p, channel, basis or "z")
    with np.errstate(divide="ignore", invalid="ignore"):
        return bare / enc
