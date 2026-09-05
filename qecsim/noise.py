"""Stochastic single-qubit Pauli noise channels.

Every channel here is a Pauli channel, so a Monte Carlo trial only needs to
*sample* a Pauli for each qubit rather than propagate a density matrix.  The
sampling is fully vectorised over the batch axis.

Channels (parameter ``p`` in ``[0, 1]``):

* ``bit_flip``       -- apply ``X`` with probability ``p``.
* ``phase_flip``     -- apply ``Z`` with probability ``p``.
* ``depolarizing``   -- apply one of ``X, Y, Z`` each with probability ``p / 3``.
"""

from __future__ import annotations

import numpy as np

from .statevector import I2, X, Z, PAULIS, apply_1q

CHANNELS = ("bit_flip", "phase_flip", "depolarizing")


def sample_pauli_indices(channel: str, p: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` Pauli indices (0=I, 1=X, 2=Y, 3=Z) for ``channel``."""
    r = rng.random(size)
    if channel == "bit_flip":
        return (r < p).astype(np.int64)  # 0 or 1 (I or X)
    if channel == "phase_flip":
        return np.where(r < p, 3, 0)  # I or Z
    if channel == "depolarizing":
        c = np.zeros(size, dtype=np.int64)
        c = np.where(r >= 1.0 - p, 1, c)          # X on the top p mass...
        c = np.where(r >= 1.0 - 2.0 * p / 3.0, 2, c)  # ...Y on the next third...
        c = np.where(r >= 1.0 - p / 3.0, 3, c)        # ...Z on the last third
        return c
    raise ValueError(f"unknown channel {channel!r}")


def apply_channel(state: np.ndarray, data_qubits, channel: str, p: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply ``channel`` independently to each qubit in ``data_qubits``."""
    b = state.shape[0]
    for q in data_qubits:
        if channel == "bit_flip":
            hit = (rng.random(b) < p).reshape(-1, 1, 1)
            ops = np.where(hit, X, I2)
        elif channel == "phase_flip":
            hit = (rng.random(b) < p).reshape(-1, 1, 1)
            ops = np.where(hit, Z, I2)
        elif channel == "depolarizing":
            ops = PAULIS[sample_pauli_indices("depolarizing", p, b, rng)]
        else:
            raise ValueError(f"unknown channel {channel!r}")
        state = apply_1q(state, ops, q)
    return state
