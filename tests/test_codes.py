import numpy as np
import pytest

from qecsim.codes import BitFlipCode, PhaseFlipCode, ShorCode, Unencoded
from qecsim.montecarlo import simulate
from qecsim.statevector import X, Y, Z
from qecsim.theory import code_logical_error


ALL = [Unencoded, BitFlipCode, PhaseFlipCode, ShorCode]


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("basis", ["z", "x"])
def test_noiseless_round_is_identity(cls, basis):
    """encode -> syndrome -> decode with no error returns the logical qubit."""
    f = cls().fidelity_with_fixed_error(basis, errors=[])
    assert f == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("q", range(3))
@pytest.mark.parametrize("pauli", [X])
def test_bitflip_corrects_single_x(q, pauli):
    assert BitFlipCode().fidelity_with_fixed_error("z", [(q, pauli)]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("q", range(3))
def test_phaseflip_corrects_single_z(q):
    assert PhaseFlipCode().fidelity_with_fixed_error("x", [(q, Z)]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("q", range(9))
@pytest.mark.parametrize("pauli", [X, Y, Z])
@pytest.mark.parametrize("basis", ["z", "x"])
def test_shor_corrects_any_single_qubit_error(q, pauli, basis):
    f = ShorCode().fidelity_with_fixed_error(basis, [(q, pauli)])
    assert f == pytest.approx(1.0, abs=1e-9)


def test_shor_fails_on_two_errors_same_block():
    # Two X errors in one block are misdecoded to X0X1X2 on that block, which
    # is a logical Z on the Shor qubit -- visible only in the X basis.
    assert ShorCode().fidelity_with_fixed_error("x", [(0, X), (1, X)]) < 0.1
    # ... and correspondingly invisible in the Z basis
    assert ShorCode().fidelity_with_fixed_error("z", [(0, X), (1, X)]) > 0.9


def test_bitflip_ignores_phase_noise_in_z_basis():
    res = simulate("bitflip_3q", "phase_flip", 0.2, n_trials=4000, seed=0)
    assert res["p_logical"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("p", [0.05, 0.1, 0.2])
def test_bitflip_matches_analytic_rate(p):
    res = simulate("bitflip_3q", "bit_flip", p, n_trials=40_000, seed=3)
    theory = float(code_logical_error(p, "bitflip_3q", "bit_flip"))
    # within ~4 sigma of the Monte Carlo estimate
    tol = 4 * max(res["p_logical_stderr"], 1e-4)
    assert abs(res["p_logical"] - theory) < tol


@pytest.mark.parametrize("p", [0.05, 0.15])
def test_phaseflip_matches_analytic_rate(p):
    res = simulate("phaseflip_3q", "phase_flip", p, n_trials=40_000, seed=4)
    theory = float(code_logical_error(p, "phaseflip_3q", "phase_flip"))
    tol = 4 * max(res["p_logical_stderr"], 1e-4)
    assert abs(res["p_logical"] - theory) < tol


def test_encoded_beats_unencoded_at_moderate_noise():
    p = 0.04
    enc = simulate("bitflip_3q", "bit_flip", p, n_trials=40_000, seed=5)
    bare = simulate("unencoded", "bit_flip", p, n_trials=40_000, seed=6)
    assert bare["p_logical"] / enc["p_logical"] > 4.0


def test_batch_size_is_statistically_consistent():
    """Batching changes how random draws map to trials, so estimates are not
    bit-identical -- but they must agree within Monte Carlo error."""
    a = simulate("bitflip_3q", "depolarizing", 0.1, n_trials=30_000, seed=7, batch_size=1)
    b = simulate("bitflip_3q", "depolarizing", 0.1, n_trials=30_000, seed=8, batch_size=4000)
    combined = (a["p_logical_stderr"] ** 2 + b["p_logical_stderr"] ** 2) ** 0.5
    assert abs(a["p_logical"] - b["p_logical"]) < 4 * max(combined, 1e-4)
