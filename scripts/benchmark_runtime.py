#!/usr/bin/env python3
"""Measure the speed-up from the batched simulation pipeline.

Runs the identical Monte Carlo workload twice -- once as a per-trial Python
loop (``batch_size=1``) and once vectorised (``batch_size=2000``) -- and
reports the wall-clock reduction.  Writes ``results/runtime.json`` for
``make_plots.py``.
"""

import argparse
import json
import os
import time

import _bootstrap  # noqa: F401

from qecsim.montecarlo import simulate


def _time(code, channel, p, n_trials, batch_size, seed=0):
    t0 = time.perf_counter()
    simulate(code, channel, p, n_trials=n_trials, seed=seed, batch_size=batch_size)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--out", default="results/runtime.json")
    args = ap.parse_args()

    configs = [
        ("bitflip_3q", "depolarizing", 0.1),
        ("phaseflip_3q", "depolarizing", 0.1),
        ("shor_9q", "depolarizing", 0.1),
    ]

    rows = []
    header = f"{'config':28s}{'loop (s)':>12s}{'batch (s)':>12s}{'speed-up':>11s}{'reduction':>12s}"
    print(header)
    print("-" * len(header))
    for code, channel, p in configs:
        loop_s = _time(code, channel, p, args.trials, batch_size=1)
        batch_s = _time(code, channel, p, args.trials, batch_size=args.batch_size)
        speedup = loop_s / batch_s
        reduction = 100.0 * (1.0 - batch_s / loop_s)
        rows.append({
            "config": f"{code.split('_')[0]}/{channel[:5]}",
            "code": code, "channel": channel, "p": p, "trials": args.trials,
            "loop_s": loop_s, "batch_s": batch_s,
            "speedup": speedup, "reduction_pct": reduction,
        })
        print(f"{code + '/' + channel:28s}{loop_s:12.2f}{batch_s:12.2f}"
              f"{speedup:10.1f}x{reduction:11.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    mean_red = sum(r["reduction_pct"] for r in rows) / len(rows)
    print(f"\nmean runtime reduction: {mean_red:.1f}%   -> {args.out}")


if __name__ == "__main__":
    main()
