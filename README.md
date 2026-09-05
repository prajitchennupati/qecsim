# Quantum Error Correction Simulator

A from-scratch NumPy state-vector simulator for **noisy quantum circuits**, plus
the three textbook stabilizer codes and a Monte Carlo harness that measures how
much they actually help.

* **Noise channels** — bit-flip, phase-flip and depolarizing, configurable
  per-qubit error rate, 1–15 qubits.
* **Codes** — 3-qubit bit-flip code, 3-qubit phase-flip code, 9-qubit Shor code,
  each with measurement-based syndrome extraction and conditioned recovery.
* **Monte Carlo** — logical error rate and logical fidelity over 10,000+ trials
  per configuration, with binomial error bars.
* **Automated benchmarking** — one command sweeps 200+ (code, channel, noise-rate)
  configurations and writes a CSV; a second command renders the figures.
* **Batch simulation pipeline** — trials are carried on a leading tensor axis and
  advanced with vectorised gate/measurement kernels, cutting sweep runtime by
  ~90% versus a per-trial Python loop.
* **Validation** — simulated rates are checked against closed-form
  error-correction formulas, and the encoders are cross-checked against
  independent Qiskit constructions.

---

## Quick start

```bash
pip install -r requirements.txt          # numpy, matplotlib, qiskit, pytest

python scripts/demo.py                    # 30-second tour
python -m pytest -q                       # 116 tests

python scripts/run_benchmark.py           # full sweep -> results/benchmark.csv
python scripts/benchmark_runtime.py       # loop vs batched -> results/runtime.json
python scripts/make_plots.py              # -> figures/*.png
```

`run_benchmark.py --quick` runs a fast, lower-statistics version.

---

## How a trial works

One Monte Carlo trial of a code is a single round of

```
prepare |0_L>  ->  encode  ->  noise channel  ->  syndrome + recovery  ->  decode  ->  read out
```

* The logical qubit is prepared in the ``|0>`` of its **protected basis**, so a
  perfect round is the identity and the ideal readout is deterministic.
* **Noise** is applied independently to each data qubit. Every channel here is a
  Pauli channel, so a trial only has to *sample* a Pauli rather than propagate a
  density matrix.
* **Syndrome extraction** uses a single ancilla, measured and reset between
  stabilizers, so an *n*-data-qubit code needs only *n + 1* qubits. Recovery is a
  Pauli conditioned on the per-trial measurement outcomes.
* **Logical error** = the decoded qubit is measured in the wrong state.
  We also report ``mean_fidelity`` = ⟨0|ρ₀|0⟩ averaged over trials, the
  sampling-noise-free complement that also captures residual decoherence.

Readout bases: the bit-flip code is read in Z (sensitive to a logical *X*), the
phase-flip code in Z after decoding (its ``|+++>`` codeword fails as a logical
bit flip), and the Shor code in a per-trial random Z/X basis so both logical
error types are seen.

## Codes

| code | qubits (data + ancilla) | corrects | stabilizers |
|------|------------------------|----------|-------------|
| `bitflip_3q`   | 3 + 1 | any single `X` | `Z0Z1`, `Z1Z2` |
| `phaseflip_3q` | 3 + 1 | any single `Z` | `X0X1`, `X1X2` |
| `shor_9q`      | 9 + 1 | any single-qubit Pauli | 6× `ZZ` (in-block) + `X0..X5`, `X3..X8` |

Every correctable single-qubit error is verified in `tests/test_codes.py` to be
corrected exactly (fidelity 1).

## Validation

For the two 3-qubit codes under the channel they are built for (and under
depolarizing noise, where the relevant per-qubit failure probability is
`2p/3`) the logical error rate has an exact closed form,
`p_L = 3q² − 2q³` (probability that ≥2 of 3 qubits fail). The simulator matches
it within Monte Carlo error across the whole noise grid — see
`tests/test_codes.py::test_bitflip_matches_analytic_rate` and the
`(analytic)` dashed curves in `figures/logical_error_*.png`. The 9-qubit Shor
code is distance-3, so its logical error rate scales as `O(p²)` and is bounded by
the two-or-more-faults probability `1 − (1−p)⁹ − 9p(1−p)⁸`.

The Qiskit cross-check (`qecsim/qiskit_reference.py`,
`tests/test_qiskit_reference.py`) rebuilds each encoder as a Qiskit circuit and
confirms the NumPy simulator produces the identical logical codewords (up to
global phase, after reconciling qubit-endianness).

## Results

From `scripts/run_benchmark.py` (10,000 trials × 224 configurations, ~2.6 min)
and `scripts/benchmark_runtime.py`. Regenerate any time; numbers below are one
representative run.

**Simulation vs. analytic logical error rate** — the 3-qubit codes agree with
`3q² − 2q³` across the grid (`q = p` for the matched channel, `q = 2p/3` for
depolarizing):

| code | channel | p | sim p_L | analytic p_L | unencoded p_L | reduction |
|---|---|---|---|---|---|---|
| `bitflip_3q`   | bit_flip     | 0.020 | 1.5e-03 ± 3.9e-04 | 1.1e-03 | 1.7e-02 | 11.1× |
| `bitflip_3q`   | bit_flip     | 0.044 | 5.2e-03 ± 7.2e-04 | 5.7e-03 | 4.4e-02 |  8.4× |
| `phaseflip_3q` | phase_flip   | 0.044 | 4.8e-03 ± 6.9e-04 | 5.7e-03 | 4.4e-02 |  9.2× |
| `bitflip_3q`   | depolarizing | 0.044 | 2.3e-03 ± 4.8e-04 | 2.6e-03 | 3.1e-02 | 13.4× |
| `shor_9q`      | depolarizing | 0.020 | 1.9e-03 ± 4.4e-04 | ≤1.3e-02 | 1.3e-02 |  6.7× |

**Peak logical-error-rate reduction vs. unencoded**, moderate-noise band
`p ∈ [0.02, 0.08]`:

| config | peak reduction | at p |
|---|---|---|
| `bitflip_3q` / bit_flip       | 10.0× | 0.026 |
| `bitflip_3q` / depolarizing   | 16.6× | 0.034 |
| `phaseflip_3q` / phase_flip   | 12.4× | 0.034 |
| `phaseflip_3q` / depolarizing | 24.0× | 0.034 |
| `shor_9q` / bit_flip          |  6.8× | 0.026 |
| `shor_9q` / depolarizing      |  3.9× | 0.026 |
| `shor_9q` / phase_flip        |  1.8× | 0.026 |

**Batch pipeline speed-up** (identical 10,000-trial workload):

| config | per-trial loop | batched | speed-up | runtime reduction |
|---|---|---|---|---|
| `bitflip_3q` / depolarizing   |  1.83 s | 0.01 s | 147× | 99% |
| `phaseflip_3q` / depolarizing |  3.11 s | 0.02 s | 154× | 99% |
| `shor_9q` / depolarizing      | 10.65 s | 3.40 s |   3× | 68% |

Figures written to `figures/`:

* `logical_error_<channel>.png` — logical vs physical error rate (log-log),
  simulation points with error bars, analytic overlay, unencoded baseline and
  the `p_L = p` break-even line.
* `fidelity_<channel>.png` — mean logical fidelity vs physical error rate.
* `improvement_factor.png` — peak logical-error-rate reduction vs unencoded in
  the moderate-noise band `p ∈ [0.02, 0.08]`.
* `runtime_comparison.png` — per-trial loop vs batched pipeline wall-clock.

## Project layout

```
qecsim/
  statevector.py      batched NumPy state-vector engine (gates, measurement)
  noise.py            bit-flip / phase-flip / depolarizing Pauli channels
  codes.py            BaseCode + Unencoded, BitFlip, PhaseFlip, Shor; MC round
  montecarlo.py       simulate(code, channel, p, ...) -> result dict
  theory.py           closed-form logical error rates for validation
  benchmark.py        multi-configuration sweep + CSV + improvement summary
  plotting.py         Matplotlib figures from a sweep
  qiskit_reference.py independent Qiskit encoders for cross-checking
scripts/
  demo.py             quick tour
  run_benchmark.py     the sweep
  benchmark_runtime.py  loop-vs-batched timing
  make_plots.py        render all figures
tests/                 116 tests (pytest)
```

## Notes and limitations

* Syndrome extraction is a single round with a perfect ancilla (no measurement
  error, no fault-tolerant repetition), so the Shor-code pseudo-threshold shown
  here is optimistic and the 3-qubit codes see no benefit against the error type
  they do not protect.
* Depolarizing results for the 3-qubit codes use the protected basis only; the
  unprotected quadrature is not corrected by construction.
