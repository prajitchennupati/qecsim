"""Cross-check the NumPy simulator against independent Qiskit constructions."""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import Operator  # noqa: E402

from qecsim.codes import CODES  # noqa: E402
from qecsim.qiskit_reference import ENCODERS, logical_codeword  # noqa: E402
from qecsim.statevector import H, X, apply_1q, apply_cx, apply_gates, zero_state  # noqa: E402


def _global_phase_overlap(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return abs(np.vdot(a, b))


def test_cx_matches_qiskit_operator():
    """apply_cx(0 -> 1) reproduces Qiskit's CX unitary once both are put in
    the same (big-endian) qubit order."""
    from qecsim.qiskit_reference import _bit_reverse_permutation

    qc = qiskit.QuantumCircuit(2)
    qc.cx(0, 1)
    rev = _bit_reverse_permutation(2)
    u_be = Operator(qc).data[np.ix_(rev, rev)]  # columns/rows -> big-endian

    for be_index, prep in [(0b00, np.eye(2, dtype=complex)), (0b10, X)]:
        got = apply_cx(apply_1q(zero_state(2, 1), prep, 0), 0, 1).reshape(-1)
        np.testing.assert_allclose(got, u_be[:, be_index], atol=1e-12)


@pytest.mark.parametrize("code", list(ENCODERS))
@pytest.mark.parametrize("inp", ["0", "1", "+", "-"])
def test_encoder_matches_qiskit_codeword(code, inp):
    ref = logical_codeword(code, inp)

    n = ENCODERS[code]().num_qubits
    prep = {"0": np.eye(2, dtype=complex), "1": X, "+": H, "-": H @ X}[inp]
    state = apply_1q(zero_state(n, 1), prep, 0)
    state = apply_gates(state, CODES[code]().encode_gates())
    got = state.reshape(-1)

    assert _global_phase_overlap(got, ref) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("code", list(ENCODERS))
def test_encode_then_decode_is_identity(code):
    enc = CODES[code]().encode_gates()
    dec = list(reversed(enc))
    rng = np.random.default_rng(0)
    # random single-qubit input via a Haar-ish 2x2 unitary
    a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = np.array([a, b]) / np.linalg.norm([a, b])
    u = np.array([[v[0], -np.conj(v[1])], [v[1], np.conj(v[0])]])

    n = ENCODERS[code]().num_qubits
    state = apply_1q(zero_state(n, 1), u, 0)
    out = apply_gates(apply_gates(state, enc), dec)
    np.testing.assert_allclose(out, state, atol=1e-12)
