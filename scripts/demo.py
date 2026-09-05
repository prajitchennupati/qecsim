#!/usr/bin/env python3
"""A 30-second tour of the simulator: single-error correction, a small sweep,
and the encoded-vs-unencoded comparison."""

import _bootstrap  # noqa: F401
import numpy as np

from qecsim.codes import BitFlipCode, ShorCode
from qecsim.montecarlo import simulate
from qecsim.statevector import X, Y, Z
from qecsim.theory import code_logical_error, unencoded_logical_error


def show_single_error_correction():
    print("== single-qubit error correction (fidelity after one round) ==")
    shor = ShorCode()
    for q in range(9):
        fids = [shor.fidelity_with_fixed_error("z", [(q, P)]) for P in (X, Y, Z)]
        print(f"  Shor, error on qubit {q}:  F(X,Y,Z) = "
              f"({fids[0]:.3f}, {fids[1]:.3f}, {fids[2]:.3f})")
    f_two_x = ShorCode().fidelity_with_fixed_error("x", [(0, X), (1, X)])
    print(f"  Shor, TWO X errors (0,1), X-basis readout: F = {f_two_x:.3f}"
          f"  (weight-2 error -> logical failure)\n")


def show_sweep():
    print("== 3-qubit bit-flip code vs unencoded, bit_flip channel ==")
    print(f"  {'p':>7s} {'unencoded':>12s} {'encoded (sim)':>15s} "
          f"{'encoded (theory)':>17s} {'reduction':>11s}")
    for p in (0.01, 0.02, 0.05, 0.1, 0.2):
        enc = simulate("bitflip_3q", "bit_flip", p, n_trials=20_000, seed=1)
        bare = simulate("unencoded", "bit_flip", p, n_trials=20_000, seed=2)
        theory = float(code_logical_error(p, "bitflip_3q", "bit_flip"))
        factor = bare["p_logical"] / max(enc["p_logical"], 1e-9)
        print(f"  {p:7.2f} {bare['p_logical']:12.4e} {enc['p_logical']:15.4e} "
              f"{theory:17.4e} {factor:10.1f}x")
    print()


def main():
    np.set_printoptions(precision=3, suppress=True)
    show_single_error_correction()
    show_sweep()
    print("run  `python scripts/run_benchmark.py`  then  "
          "`python scripts/make_plots.py`  for the full study.")


if __name__ == "__main__":
    main()
