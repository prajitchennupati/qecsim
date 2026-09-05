"""Batched pure-state state-vector simulator built on NumPy.

The state is stored as a complex128 tensor of shape ``(B,) + (2,) * n`` where
``B`` is the number of independent Monte Carlo trials carried in parallel
(the "batch") and each remaining axis is one qubit.  Qubit ``q`` lives on
tensor axis ``q + 1``; qubit 0 is the most significant bit of the flat index.

Carrying a batch axis is what turns a per-trial Python loop into a handful of
vectorised BLAS calls -- see :mod:`qecsim.montecarlo` and the runtime
benchmark in ``scripts/benchmark_runtime.py``.
"""

from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

#: Stack of the single-qubit Paulis indexed 0->I, 1->X, 2->Y, 3->Z.
PAULIS = np.stack([I2, X, Y, Z])


def zero_state(n: int, batch: int = 1) -> np.ndarray:
    """Return ``batch`` copies of the n-qubit ground state ``|0...0>``."""
    psi = np.zeros((batch,) + (2,) * n, dtype=np.complex128)
    psi[(slice(None),) + (0,) * n] = 1.0
    return psi


def n_qubits(state: np.ndarray) -> int:
    """Number of qubit axes in ``state`` (its rank minus the batch axis)."""
    return state.ndim - 1


def apply_1q(state: np.ndarray, u: np.ndarray, q: int) -> np.ndarray:
    """Apply a single-qubit operator to qubit ``q`` of every trial.

    ``u`` is either a shared ``(2, 2)`` matrix or a per-trial ``(B, 2, 2)``
    stack (used for measurement-conditioned recovery gates).
    """
    ax = q + 1
    s = np.moveaxis(state, ax, 1)
    if u.ndim == 2:
        s = np.einsum("ij,bj...->bi...", u, s, optimize=True)
    else:
        s = np.einsum("bij,bj...->bi...", u, s, optimize=True)
    return np.moveaxis(s, 1, ax)


def apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply CNOT(control -> target) to every trial."""
    ac, at = control + 1, target + 1
    s = np.moveaxis(state, [ac, at], [1, 2])
    out = s.copy()
    out[:, 1] = s[:, 1][:, ::-1]  # control == 1: swap the target amplitudes
    return np.moveaxis(out, [1, 2], [ac, at])


def apply_cz(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply CZ(control, target) to every trial."""
    ac, at = control + 1, target + 1
    s = np.moveaxis(state, [ac, at], [1, 2]).copy()
    s[:, 1, 1] *= -1.0
    return np.moveaxis(s, [1, 2], [ac, at])


_1Q = {"h": H, "x": X, "y": Y, "z": Z, "s": S}


def apply_gates(state: np.ndarray, gates) -> np.ndarray:
    """Apply an ordered iterable of gate tuples.

    Supported tuples: ``("h"|"x"|"y"|"z"|"s", q)``, ``("cx", c, t)``,
    ``("cz", c, t)``.
    """
    for g in gates:
        kind = g[0]
        if kind in _1Q:
            state = apply_1q(state, _1Q[kind], g[1])
        elif kind == "cx":
            state = apply_cx(state, g[1], g[2])
        elif kind == "cz":
            state = apply_cz(state, g[1], g[2])
        else:  # pragma: no cover - guards against typos in gate lists
            raise ValueError(f"unknown gate {kind!r}")
    return state


def measure(state: np.ndarray, q: int, rng: np.random.Generator):
    """Projectively measure qubit ``q`` in the Z basis for every trial.

    Returns ``(collapsed_state, outcomes)`` where ``outcomes`` is an
    ``int8`` array of shape ``(B,)``.  Each trial is sampled and collapsed
    independently, so different trials may take different branches.
    """
    n = n_qubits(state)
    ax = q + 1
    s = np.moveaxis(state, ax, 1)
    p1 = np.sum(np.abs(s[:, 1]) ** 2, axis=tuple(range(1, n)))
    outcome = (rng.random(s.shape[0]) < p1).astype(np.int8)
    s = s.copy()
    s[outcome == 0, 1] = 0.0
    s[outcome == 1, 0] = 0.0
    norm = np.sqrt(np.sum(np.abs(s) ** 2, axis=tuple(range(1, n + 1))))
    norm = np.where(norm < 1e-15, 1.0, norm)
    s = s / norm.reshape((-1,) + (1,) * n)
    return np.moveaxis(s, 1, ax), outcome


def conditional_pauli(state: np.ndarray, q: int, cond, pauli: np.ndarray) -> np.ndarray:
    """Apply ``pauli`` to qubit ``q`` only in the trials where ``cond`` is true."""
    cond = np.asarray(cond).reshape(-1, 1, 1).astype(bool)
    ops = np.where(cond, pauli, I2)
    return apply_1q(state, ops, q)


def prob_zero(state: np.ndarray, q: int) -> np.ndarray:
    """Per-trial probability of measuring ``0`` on qubit ``q`` (no collapse)."""
    n = n_qubits(state)
    s = np.moveaxis(state, q + 1, 1)
    return np.sum(np.abs(s[:, 0]) ** 2, axis=tuple(range(1, n)))
