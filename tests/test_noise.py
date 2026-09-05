import numpy as np
import pytest

from qecsim.noise import apply_channel, sample_pauli_indices
from qecsim.statevector import prob_zero, zero_state


@pytest.mark.parametrize("channel", ["bit_flip", "phase_flip", "depolarizing"])
def test_zero_rate_is_identity(channel):
    rng = np.random.default_rng(0)
    psi = zero_state(3, 64)
    out = apply_channel(psi.copy(), [0, 1, 2], channel, 0.0, rng)
    np.testing.assert_allclose(out, psi, atol=1e-12)


def test_bit_flip_frequency_matches_rate():
    rng = np.random.default_rng(0)
    psi = zero_state(1, 40_000)
    out = apply_channel(psi, [0], "bit_flip", 0.2, rng)
    # fraction of trials now in |1>
    flipped = 1.0 - prob_zero(out, 0)
    assert abs(flipped.mean() - 0.2) < 0.01


def test_phase_flip_leaves_populations_untouched():
    rng = np.random.default_rng(0)
    psi = zero_state(1, 5000)
    out = apply_channel(psi, [0], "phase_flip", 0.5, rng)
    np.testing.assert_allclose(prob_zero(out, 0), 1.0, atol=1e-12)


def test_depolarizing_pauli_distribution():
    rng = np.random.default_rng(1)
    idx = sample_pauli_indices("depolarizing", 0.3, 200_000, rng)
    counts = np.bincount(idx, minlength=4) / idx.size
    np.testing.assert_allclose(counts, [0.7, 0.1, 0.1, 0.1], atol=0.005)


def test_bit_flip_only_samples_i_and_x():
    rng = np.random.default_rng(2)
    idx = sample_pauli_indices("bit_flip", 0.4, 10_000, rng)
    assert set(np.unique(idx)).issubset({0, 1})
