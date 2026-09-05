import numpy as np
import pytest

from qecsim.theory import (
    _rep3,
    _two_or_more,
    code_logical_error,
    improvement_factor,
    unencoded_logical_error,
)


def test_rep3_leading_order():
    p = 1e-4
    assert _rep3(p) == pytest.approx(3 * p**2, rel=1e-3)


def test_rep3_endpoints():
    assert _rep3(0.0) == 0.0
    assert _rep3(0.5) == pytest.approx(0.5)  # 3/4 - 1/4
    assert _rep3(1.0) == pytest.approx(1.0)


def test_two_or_more_matches_binomial():
    p, n = 0.1, 9
    # P(>=2) = 1 - P(0) - P(1)
    expect = 1 - (0.9 ** 9) - 9 * 0.1 * (0.9 ** 8)
    assert _two_or_more(p, n) == pytest.approx(expect)


def test_unencoded_depolarizing_is_two_thirds_p():
    p = np.array([0.01, 0.1, 0.3])
    np.testing.assert_allclose(unencoded_logical_error(p, "depolarizing"), 2 * p / 3)


def test_unencoded_phase_flip_invisible_in_z_basis():
    assert unencoded_logical_error(0.2, "phase_flip", basis="z") == 0.0
    assert unencoded_logical_error(0.2, "phase_flip", basis="x") == pytest.approx(0.2)


def test_code_curves_are_monotonic_increasing():
    p = np.linspace(0.001, 0.3, 40)
    for code, channel in [("bitflip_3q", "bit_flip"),
                          ("phaseflip_3q", "phase_flip"),
                          ("shor_9q", "depolarizing")]:
        y = np.asarray(code_logical_error(p, code, channel))
        assert np.all(np.diff(y) >= -1e-12)


def test_improvement_factor_large_at_small_p():
    # 3-qubit code: p / (3 p^2) = 1 / (3p) -> large as p -> 0
    assert improvement_factor(0.01, "bitflip_3q", "bit_flip") > 20


def test_unknown_combo_returns_nan():
    out = np.asarray(code_logical_error(np.array([0.1]), "shor_9q", "nonsense"))
    assert np.isnan(out).all()
