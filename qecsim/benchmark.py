"""Automated sweep over many (code, channel, noise-rate) configurations.

For every entry in :data:`CODE_CHANNEL_PAIRS` the sweep runs the code and a
matched unencoded baseline across a grid of physical error rates, and records
one row per simulation.  Code rows additionally carry the paired baseline
number and the resulting logical-error-rate reduction.  The default grid
produces well over 100 configurations.

A pair is ``(code, channel, code_basis, baseline_basis)``.  The two bases can
differ: the phase-flip code stores its logical qubit as ``|+++>`` and its
failure shows up as a *bit* flip after decoding, so it is read in the Z basis
while the phase-noise-exposed bare qubit it is compared against is ``|+>``
read in the X basis.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from .montecarlo import simulate
from .theory import code_logical_error

#: 16-point log-spaced grid of physical error rates.
DEFAULT_P_GRID = np.round(np.geomspace(0.005, 0.3, 16), 6)

#: (code, channel, code readout basis, baseline readout basis).
CODE_CHANNEL_PAIRS = [
    ("bitflip_3q", "bit_flip", "z", "z"),
    ("bitflip_3q", "depolarizing", "z", "z"),
    ("phaseflip_3q", "phase_flip", "z", "x"),
    ("phaseflip_3q", "depolarizing", "z", "x"),
    ("shor_9q", "depolarizing", "random", "random"),
    ("shor_9q", "bit_flip", "random", "random"),
    ("shor_9q", "phase_flip", "random", "random"),
]

CSV_FIELDS = [
    "role", "code", "channel", "basis", "p", "n_trials", "p_logical",
    "p_logical_stderr", "mean_fidelity", "p_logical_theory",
    "baseline_p_logical", "improvement", "runtime_s",
]


def run_sweep(p_grid=None, n_trials=10_000, seed=0, batch_size=2000, progress=True):
    """Run the full benchmark and return a list of result dicts."""
    p_grid = DEFAULT_P_GRID if p_grid is None else np.asarray(p_grid, dtype=float)
    rows = []
    baseline_cache = {}
    seed_counter = 0
    baseline_seed = 0
    t_start = time.perf_counter()
    n_sims = len(CODE_CHANNEL_PAIRS) * len(p_grid)

    for code, channel, cbasis, bbasis in CODE_CHANNEL_PAIRS:
        for p in p_grid:
            p = float(p)

            key = (channel, bbasis, p)
            if key not in baseline_cache:
                base = simulate("unencoded", channel, p, n_trials=n_trials,
                                seed=seed + 100_000 + baseline_seed,
                                basis=bbasis, batch_size=batch_size)
                baseline_seed += 1
                base["role"] = "baseline"
                base["p_logical_theory"] = float(
                    np.asarray(code_logical_error(p, "unencoded", channel, bbasis)))
                base["baseline_p_logical"] = ""
                base["improvement"] = ""
                base["runtime_s"] = 0.0
                baseline_cache[key] = base
                rows.append(base)

            t0 = time.perf_counter()
            res = simulate(code, channel, p, n_trials=n_trials,
                           seed=seed + seed_counter, basis=cbasis,
                           batch_size=batch_size)
            seed_counter += 1
            res["role"] = "code"
            res["runtime_s"] = time.perf_counter() - t0
            res["p_logical_theory"] = float(
                np.asarray(code_logical_error(p, code, channel, cbasis)))
            bl = baseline_cache[key]["p_logical"]
            res["baseline_p_logical"] = bl
            res["improvement"] = (bl / res["p_logical"]) if res["p_logical"] > 0 else float("inf")
            rows.append(res)

            if progress:
                print(f"[{seed_counter:3d}/{n_sims}] {code:13s} {channel:13s} "
                      f"p={p:.5f}  p_L={res['p_logical']:.3e}"
                      f" +/- {res['p_logical_stderr']:.1e}   "
                      f"baseline={bl:.3e}  x{res['improvement']:.1f}"
                      f"  ({res['runtime_s']:.2f}s)")

    if progress:
        print(f"\n{len(rows)} rows from {n_sims} code sims + "
              f"{len(baseline_cache)} baselines in "
              f"{time.perf_counter() - t_start:.1f}s")
    return rows


def write_csv(rows, path):
    """Write sweep rows to ``path`` as CSV."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def read_csv(path):
    """Read a sweep CSV back into a list of dicts with numeric fields cast."""
    numeric = {"p", "n_trials", "p_logical", "p_logical_stderr", "mean_fidelity",
               "p_logical_theory", "baseline_p_logical", "improvement", "runtime_s"}
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in numeric:
                if r.get(k) not in (None, ""):
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def summarize_improvement(rows, p_lo=0.02, p_hi=0.08):
    """Peak logical-error-rate reduction versus the unencoded baseline in the
    moderate-noise band ``[p_lo, p_hi]`` for each (code, channel)."""
    grouped = {}
    for r in rows:
        if r.get("role") != "code":
            continue
        grouped.setdefault((r["code"], r["channel"]), []).append(r)

    out = []
    for (code, channel), items in sorted(grouped.items()):
        best = None
        for r in items:
            if not (p_lo <= r["p"] <= p_hi):
                continue
            bl = r.get("baseline_p_logical")
            if not bl or r["p_logical"] <= 0:
                continue
            factor = bl / r["p_logical"]
            if best is None or factor > best["factor"]:
                best = {"code": code, "channel": channel, "p": r["p"],
                        "factor": factor, "p_logical": r["p_logical"],
                        "unencoded": bl}
        if best:
            out.append(best)
    return out
