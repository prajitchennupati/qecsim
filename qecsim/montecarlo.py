"""Thin front door to the Monte Carlo engine in :mod:`qecsim.codes`."""

from __future__ import annotations

import numpy as np

from .codes import CODES


def simulate(code, channel, p, n_trials=10_000, seed=0, basis=None, batch_size=2000):
    """Estimate the logical error rate of ``code`` under ``channel`` at rate ``p``.

    Parameters
    ----------
    code : str
        One of :data:`qecsim.codes.CODES` (``unencoded``, ``bitflip_3q``,
        ``phaseflip_3q``, ``shor_9q``).
    channel : str
        ``bit_flip``, ``phase_flip`` or ``depolarizing``.
    p : float
        Physical error rate per qubit.
    n_trials : int
        Number of Monte Carlo rounds.
    seed : int
        Seed for the trial's ``numpy`` random generator.
    basis : str, optional
        Override the code's readout basis (``z``, ``x`` or ``random``).
    batch_size : int
        Trials evaluated per vectorised sweep.  ``1`` reproduces a plain
        per-trial Python loop; a few thousand is dramatically faster.

    Returns
    -------
    dict
        Keys: ``code, channel, basis, p, n_trials, p_logical,
        p_logical_stderr, mean_fidelity``.
    """
    if code not in CODES:
        raise KeyError(f"unknown code {code!r}; choose from {sorted(CODES)}")
    rng = np.random.default_rng(seed)
    return CODES[code]().run_trials(
        channel, p, n_trials=n_trials, rng=rng, basis=basis, batch_size=batch_size
    )
