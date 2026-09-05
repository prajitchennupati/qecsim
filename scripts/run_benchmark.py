#!/usr/bin/env python3
"""Run the automated noise-configuration sweep and write results to CSV.

Examples
--------
    python scripts/run_benchmark.py                       # full 10k-trial sweep
    python scripts/run_benchmark.py --quick               # fast smoke run
    python scripts/run_benchmark.py --trials 20000 --out results/big.csv
"""

import argparse
import os

import _bootstrap  # noqa: F401  (sys.path setup)
import numpy as np

from qecsim.benchmark import run_sweep, summarize_improvement, write_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10_000,
                    help="Monte Carlo trials per configuration (default 10000)")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", type=int, default=16,
                    help="number of log-spaced physical error rates")
    ap.add_argument("--pmin", type=float, default=0.005)
    ap.add_argument("--pmax", type=float, default=0.3)
    ap.add_argument("--quick", action="store_true",
                    help="shortcut for --trials 1500 --points 8")
    ap.add_argument("--out", default="results/benchmark.csv")
    args = ap.parse_args()

    if args.quick:
        args.trials, args.points = 1500, 8

    p_grid = np.round(np.geomspace(args.pmin, args.pmax, args.points), 6)
    rows = run_sweep(p_grid=p_grid, n_trials=args.trials, seed=args.seed,
                     batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    print("\npeak logical error-rate reduction vs unencoded (p in [0.02, 0.08]):")
    for s in summarize_improvement(rows):
        print(f"  {s['code']:13s} {s['channel']:13s} "
              f"{s['factor']:5.1f}x  (p={s['p']:.4f}: "
              f"{s['unencoded']:.3e} -> {s['p_logical']:.3e})")


if __name__ == "__main__":
    main()
