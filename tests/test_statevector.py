import numpy as np
import pytest

from qecsim.statevector import (
    H,
    X,
    apply_1q,
    apply_cx,
    apply_cz,
    measure,
    n_qubits,
    prob_zero,
    zero_state,
)


def test_zero_state_shape_and_norm():
    psi = zero_state(4, batch=3)
    assert psi.shape == (3, 2, 2, 2, 2)
    assert n_qubits(psi) == 4
    flat = psi.reshape(3, -1)
    np.testing.assert_allclose(np.sum(np.abs(flat) ** 2, axis=1), 1.0)
    assert flat[0, 0] == 1.0  # only |0000> populated


def test_hadamard_creates_superposition():
    psi = apply_1q(zero_state(1, 1), H, 0)
    np.testing.assert_allclose(psi.reshape(-1), [1 / np.sqrt(2)] * 2)


def test_bell_state_via_h_then_cx():
    psi = apply_cx(apply_1q(zero_state(2, 1), H, 0), 0, 1)
    amp = psi.reshape(-1)
    np.testing.assert_allclose(amp, [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], atol=1e-12)


def test_cx_flips_target_when_control_one():
    psi = apply_cx(apply_1q(zero_state(2, 1), X, 0), 0, 1)
    amp = psi.reshape(-1)
    np.testing.assert_allclose(amp, [0, 0, 0, 1], atol=1e-12)  # |11>


def test_cz_phase():
    psi = zero_state(2, 1)
    psi = apply_1q(apply_1q(psi, H, 0), H, 1)  # uniform superposition
    psi = apply_cz(psi, 0, 1)
    np.testing.assert_allclose(psi.reshape(-1), [0.5, 0.5, 0.5, -0.5], atol=1e-12)


def test_measure_statistics_on_plus_state():
    rng = np.random.default_rng(0)
    psi = apply_1q(zero_state(1, 20_000), H, 0)
    _, outcomes = measure(psi, 0, rng)
    assert abs(outcomes.mean() - 0.5) < 0.02


def test_measure_collapses_and_renormalizes():
    rng = np.random.default_rng(1)
    psi = apply_1q(zero_state(1, 500), H, 0)
    collapsed, outcomes = measure(psi, 0, rng)
    norms = np.sum(np.abs(collapsed.reshape(500, -1)) ** 2, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    # after collapse prob_zero is exactly 0 or 1 and matches the outcome
    pz = prob_zero(collapsed, 0)
    np.testing.assert_allclose(pz, 1.0 - outcomes, atol=1e-12)


def test_per_trial_operator_broadcasts():
    psi = zero_state(1, 2)
    ops = np.stack([np.eye(2, dtype=complex), X])  # trial 0: I, trial 1: X
    out = apply_1q(psi, ops, 0).reshape(2, -1)
    np.testing.assert_allclose(out[0], [1, 0])
    np.testing.assert_allclose(out[1], [0, 1])
