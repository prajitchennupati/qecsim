#!/usr/bin/env python3
"""Generate all figures from a benchmark CSV.

    python scripts/make_plots.py --in results/benchmark.csv --outdir figures
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401

from qecsim.benchmark import read_csv, summarize_improvement
from qecsim.plotting import (
    plot_fidelity,
    plot_improvement,
    plot_logical_error,
    plot_runtime,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="results/benchmark.csv")
    ap.add_argument("--runtime", default="results/runtime.json",
                    help="optional runtime.json from benchmark_runtime.py")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    rows = read_csv(args.inp)
    os.makedirs(args.outdir, exist_ok=True)
    channels = sorted({r["channel"] for r in rows})

    written = []
    for ch in channels:
        p1 = os.path.join(args.outdir, f"logical_error_{ch}.png")
        p2 = os.path.join(args.outdir, f"fidelity_{ch}.png")
        plot_logical_error(rows, ch, p1)
        plot_fidelity(rows, ch, p2)
        written += [p1, p2]

    imp = os.path.join(args.outdir, "improvement_factor.png")
    plot_improvement(summarize_improvement(rows), imp)
    written.append(imp)

    if os.path.exists(args.runtime):
        with open(args.runtime) as fh:
            rt = json.load(fh)
        p = os.path.join(args.outdir, "runtime_comparison.png")
        plot_runtime(rt, p)
        written.append(p)

    print("wrote:")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
