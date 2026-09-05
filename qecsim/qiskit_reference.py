"""Independent Qiskit constructions of the encoders, used to cross-check the
NumPy simulator.

Qiskit orders its statevector little-endian (qubit 0 is the least significant
bit); :mod:`qecsim.statevector` is big-endian.  :func:`to_qecsim_order`
bridges the two so amplitudes can be compared directly.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def bitflip_encoder() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="bitflip_encode")
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


def phaseflip_encoder() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="phaseflip_encode")
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


def shor_encoder() -> QuantumCircuit:
    qc = QuantumCircuit(9, name="shor_encode")
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)
    for head in (0, 3, 6):
        qc.cx(head, head + 1)
        qc.cx(head, head + 2)
    return qc


ENCODERS = {
    "bitflip_3q": bitflip_encoder,
    "phaseflip_3q": phaseflip_encoder,
    "shor_9q": shor_encoder,
}


def _bit_reverse_permutation(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    out = np.zeros_like(idx)
    for i in range(2 ** n):
        b, r = i, 0
        for _ in range(n):
            r = (r << 1) | (b & 1)
            b >>= 1
        out[i] = r
    return out


def to_qecsim_order(statevector: Statevector) -> np.ndarray:
    """Reorder a Qiskit little-endian statevector into big-endian layout."""
    data = np.asarray(statevector.data, dtype=np.complex128)
    n = int(np.log2(data.size))
    return data[_bit_reverse_permutation(n)]


def logical_codeword(code: str, input_label: str = "0") -> np.ndarray:
    """Big-endian statevector of ``code`` encoding the single-qubit state
    given by ``input_label`` (``"0"``, ``"1"``, ``"+"`` or ``"-"``)."""
    enc = ENCODERS[code]()
    label = "0" * (enc.num_qubits - 1) + input_label
    sv = Statevector.from_label(label).evolve(enc)
    return to_qecsim_order(sv)
