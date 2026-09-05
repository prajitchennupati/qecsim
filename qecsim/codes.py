"""Quantum error-correcting codes and the Monte Carlo round that exercises them.

Each code implements one round of

    prepare logical state -> encode -> noise -> syndrome + recovery -> decode

and then reads the logical qubit out in its protected basis.  Syndrome
extraction is done with a *single* re-used ancilla (measured and reset
between stabilizers) so the state stays small: an ``n``-data-qubit code needs
only ``n + 1`` qubits.

Logical-error metric
--------------------
After a perfect round the logical qubit is returned to its input state, so we
prepare the ``|0>`` of the protected basis, undo that preparation after
decoding, and measure qubit 0.  Outcome ``1`` is a logical error.  We also
report ``mean_fidelity`` = ``<0| rho_0 |0>`` averaged over trials, which is
the sampling-noise-free complement of the logical error probability and also
picks up any residual decoherence.

* ``bitflip_3q``   is read in the Z basis  (sensitive to logical ``X``).
* ``phaseflip_3q`` is read in the X basis  (sensitive to logical ``Z``).
* ``shor_9q``      picks the Z or X basis at random per trial (both error types).
"""

from __future__ import annotations

import numpy as np

from .noise import apply_channel
from .statevector import (
    H,
    I2,
    X,
    Z,
    apply_1q,
    apply_cx,
    apply_gates,
    conditional_pauli,
    measure,
    prob_zero,
    zero_state,
)

# Preparation unitary for the logical "0" of each readout basis.  Both are
# Hermitian and unitary, hence self-inverse, so the same matrix undoes the
# preparation after decoding.
_BASIS_PREP = np.stack([I2, H])  # index 0 -> Z basis, 1 -> X basis


def _bitflip_syndrome(state, rng, anc, block=(0, 1, 2)):
    """Measure ``Z Z`` on the two neighbouring pairs of ``block`` and apply the
    minimum-weight ``X`` correction.  Shared by the bit-flip code and by each
    inner block of the Shor code."""
    a, b, c = block
    state = apply_cx(state, a, anc)
    state = apply_cx(state, b, anc)
    state, s0 = measure(state, anc, rng)
    state = conditional_pauli(state, anc, s0, X)  # reset ancilla to |0>
    state = apply_cx(state, b, anc)
    state = apply_cx(state, c, anc)
    state, s1 = measure(state, anc, rng)
    state = conditional_pauli(state, anc, s1, X)

    f0, f1 = s0.astype(bool), s1.astype(bool)
    state = conditional_pauli(state, a, f0 & ~f1, X)
    state = conditional_pauli(state, b, f0 & f1, X)
    state = conditional_pauli(state, c, ~f0 & f1, X)
    return state


class BaseCode:
    """Common Monte Carlo driver; subclasses supply the circuit pieces."""

    name = "base"
    n_data = 1
    default_basis = "z"  # "z", "x", or "random"

    # --- circuit pieces overridden by subclasses -------------------------
    def encode_gates(self):
        return []

    def n_ancilla(self):
        return 0

    def syndrome_recover(self, state, rng):
        return state

    # --- shared machinery ----------------------------------------------
    def _basis_indices(self, size, basis, rng):
        if basis == "z":
            return np.zeros(size, dtype=np.int64)
        if basis == "x":
            return np.ones(size, dtype=np.int64)
        if basis == "random":
            return rng.integers(0, 2, size=size)
        raise ValueError(f"unknown basis {basis!r}")

    def run_trials(self, channel, p, n_trials=10_000, rng=None, basis=None,
                   batch_size=2000):
        """Run ``n_trials`` independent rounds and return a result dict."""
        rng = np.random.default_rng() if rng is None else rng
        basis = self.default_basis if basis is None else basis
        enc = self.encode_gates()
        dec = list(reversed(enc))  # every gate we use is self-inverse
        data_qubits = list(range(self.n_data))
        n_total = self.n_data + self.n_ancilla()

        n_err = 0
        fid_sum = 0.0
        done = 0
        while done < n_trials:
            b = int(min(batch_size, n_trials - done))
            bidx = self._basis_indices(b, basis, rng)
            prep = _BASIS_PREP[bidx]  # (b, 2, 2)

            state = zero_state(n_total, b)
            state = apply_1q(state, prep, 0)
            state = apply_gates(state, enc)
            state = apply_channel(state, data_qubits, channel, p, rng)
            state = self.syndrome_recover(state, rng)
            state = apply_gates(state, dec)
            state = apply_1q(state, prep, 0)  # undo preparation

            fid_sum += float(np.sum(prob_zero(state, 0)))
            state, outcome = measure(state, 0, rng)
            n_err += int(np.count_nonzero(outcome))
            done += b

        p_log = n_err / n_trials
        stderr = float(np.sqrt(max(p_log * (1.0 - p_log), 0.0) / n_trials))
        return {
            "code": self.name,
            "channel": channel,
            "basis": basis,
            "p": float(p),
            "n_trials": int(n_trials),
            "p_logical": p_log,
            "p_logical_stderr": stderr,
            "mean_fidelity": fid_sum / n_trials,
        }

    def fidelity_with_fixed_error(self, basis, errors, rng=None):
        """Diagnostic: run one noiseless round except for a fixed list of
        ``(qubit, pauli_matrix)`` errors, and return ``<0| rho_0 |0>``.

        Used by the tests to check that every correctable single-qubit error
        is in fact corrected.
        """
        rng = np.random.default_rng(0) if rng is None else rng
        enc = self.encode_gates()
        dec = list(reversed(enc))
        prep = _BASIS_PREP[0 if basis == "z" else 1]
        state = zero_state(self.n_data + self.n_ancilla(), 1)
        state = apply_1q(state, prep, 0)
        state = apply_gates(state, enc)
        for q, pauli in errors:
            state = apply_1q(state, pauli, q)
        state = self.syndrome_recover(state, rng)
        state = apply_gates(state, dec)
        state = apply_1q(state, prep, 0)
        return float(prob_zero(state, 0)[0])


class Unencoded(BaseCode):
    """A single bare qubit -- the baseline the codes are measured against."""

    name = "unencoded"
    n_data = 1
    default_basis = "z"


class BitFlipCode(BaseCode):
    """3-qubit repetition code.  Corrects any single ``X`` error."""

    name = "bitflip_3q"
    n_data = 3
    default_basis = "z"

    def n_ancilla(self):
        return 1

    def encode_gates(self):
        return [("cx", 0, 1), ("cx", 0, 2)]

    def syndrome_recover(self, state, rng):
        return _bitflip_syndrome(state, rng, anc=self.n_data, block=(0, 1, 2))


class PhaseFlipCode(BaseCode):
    """3-qubit repetition code in the Hadamard basis.  Corrects any single
    ``Z`` error (it is :class:`BitFlipCode` conjugated by ``H`` on every data
    qubit)."""

    name = "phaseflip_3q"
    n_data = 3
    # encode(|0>) = |+++> = |0_L>; an uncorrectable (weight >= 2) phase error
    # is misdecoded as the other codeword, i.e. a logical *bit* flip, which is
    # what a Z-basis readout of the decoded qubit detects.
    default_basis = "z"

    def n_ancilla(self):
        return 1

    def encode_gates(self):
        return [("cx", 0, 1), ("cx", 0, 2), ("h", 0), ("h", 1), ("h", 2)]

    def syndrome_recover(self, state, rng):
        for q in (0, 1, 2):
            state = apply_1q(state, H, q)  # rotate into the bit-flip frame
        state = _bitflip_syndrome(state, rng, anc=self.n_data, block=(0, 1, 2))
        for q in (0, 1, 2):
            state = apply_1q(state, H, q)
        return state


class ShorCode(BaseCode):
    """9-qubit Shor code: a phase-flip code whose three qubits are each a
    bit-flip codeword.  Corrects an arbitrary single-qubit error."""

    name = "shor_9q"
    n_data = 9
    default_basis = "random"

    def n_ancilla(self):
        return 1

    def encode_gates(self):
        gates = [("cx", 0, 3), ("cx", 0, 6), ("h", 0), ("h", 3), ("h", 6)]
        for head in (0, 3, 6):
            gates += [("cx", head, head + 1), ("cx", head, head + 2)]
        return gates

    def syndrome_recover(self, state, rng):
        anc = self.n_data  # qubit 9

        # 1. correct bit flips inside each block
        for block in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
            state = _bitflip_syndrome(state, rng, anc, block)

        # 2. correct a phase flip between blocks.  Measure the X-type
        #    stabilizers X0..X5 and X3..X8 via the standard ancilla circuit
        #    H(anc) - CX(anc -> support) - H(anc) - measure.
        outcomes = []
        for support in (range(0, 6), range(3, 9)):
            state = apply_1q(state, H, anc)
            for i in support:
                state = apply_cx(state, anc, i)
            state = apply_1q(state, H, anc)
            state, t = measure(state, anc, rng)
            state = conditional_pauli(state, anc, t, X)
            outcomes.append(t.astype(bool))

        t0, t1 = outcomes
        state = conditional_pauli(state, 0, t0 & ~t1, Z)  # phase error in block 0
        state = conditional_pauli(state, 3, t0 & t1, Z)   # ... block 1
        state = conditional_pauli(state, 6, ~t0 & t1, Z)  # ... block 2
        return state


CODES = {
    cls.name: cls for cls in (Unencoded, BitFlipCode, PhaseFlipCode, ShorCode)
}
