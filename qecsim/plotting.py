"""Matplotlib figures built from a benchmark sweep (see :mod:`qecsim.benchmark`)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_CODE_COLOR = {
    "unencoded": "#9aa0a6",
    "bitflip_3q": "#1f77b4",
    "phaseflip_3q": "#2ca02c",
    "shor_9q": "#d62728",
}
_CODE_LABEL = {
    "bitflip_3q": "3-qubit bit-flip",
    "phaseflip_3q": "3-qubit phase-flip",
    "shor_9q": "9-qubit Shor",
}


def _code_rows(rows, channel):
    out = {}
    for r in rows:
        if r.get("role") == "code" and r["channel"] == channel:
            out.setdefault(r["code"], []).append(r)
    return {c: sorted(v, key=lambda r: r["p"]) for c, v in out.items()}


def plot_logical_error(rows, channel, path):
    """Logical vs physical error rate for every code on ``channel``, with the
    analytic curve and the paired unencoded baseline overlaid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    series = _code_rows(rows, channel)
    all_p = sorted({r["p"] for v in series.values() for r in v})

    floor = 3e-5
    baseline_drawn = False
    for code, pts in series.items():
        p = np.array([r["p"] for r in pts])
        pl = np.clip([r["p_logical"] for r in pts], floor, 1)
        err = np.array([r["p_logical_stderr"] for r in pts])
        # asymmetric bars so the lower whisker never crosses zero on a log axis
        yerr = np.vstack([np.minimum(err, pl * 0.9), err])
        color = _CODE_COLOR[code]
        ax.errorbar(p, pl, yerr=yerr, fmt="o", ms=4, color=color, capsize=2,
                    label=f"{_CODE_LABEL[code]} (sim)")

        th = np.array([r.get("p_logical_theory", np.nan) for r in pts], dtype=float)
        if np.isfinite(th).any():
            ax.plot(p, np.clip(th, floor, 1), "--", lw=1.4, color=color,
                    label=f"{_CODE_LABEL[code]} (analytic)")

        bl = np.array([r.get("baseline_p_logical", np.nan) for r in pts], dtype=float)
        if np.isfinite(bl).any() and not baseline_drawn:
            ax.plot(p, np.clip(bl, floor, 1), "-", lw=1.4, color="#9aa0a6",
                    alpha=0.8, label="unencoded baseline (sim)")
            baseline_drawn = True

    if all_p:
        lo = np.array([min(all_p), max(all_p)])
        ax.plot(lo, lo, ":", color="k", lw=1, label="break-even ($p_L = p$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor * 0.7)
    ax.set_xlabel("physical error rate  $p$")
    ax.set_ylabel("logical error rate  $p_L$")
    ax.set_title(f"Logical error rate -- {channel} channel")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fidelity(rows, channel, path):
    """Mean logical-state fidelity vs physical error rate."""
    fig, ax = plt.subplots(figsize=(7, 5))
    series = _code_rows(rows, channel)
    for code, pts in series.items():
        p = np.array([r["p"] for r in pts])
        fid = np.array([r["mean_fidelity"] for r in pts])
        ax.plot(p, fid, "o-", ms=4, color=_CODE_COLOR[code],
                label=_CODE_LABEL[code])
    # one representative baseline fidelity line
    base = sorted((r for r in rows if r.get("role") == "baseline"
                   and r["channel"] == channel), key=lambda r: r["p"])
    if base:
        ax.plot([r["p"] for r in base], [r["mean_fidelity"] for r in base],
                "s--", ms=3, color="#9aa0a6", label="unencoded")
    ax.set_xscale("log")
    ax.set_xlabel("physical error rate  $p$")
    ax.set_ylabel("mean logical fidelity")
    ax.set_title(f"Logical fidelity -- {channel} channel")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_improvement(summary, path):
    """Bar chart of the peak logical-error-rate reduction per configuration."""
    if not summary:
        return
    labels = [f"{s['code'].replace('_3q', '').replace('_9q', '')}\n{s['channel']}"
              for s in summary]
    factors = [s["factor"] for s in summary]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), factors, color="#1f77b4", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    for bar, s in zip(bars, summary):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s['factor']:.1f}x\n@p={s['p']:.3f}", ha="center",
                va="bottom", fontsize=8)
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_ylabel(r"peak  $p_L^{\mathrm{unencoded}} / p_L^{\mathrm{encoded}}$")
    ax.set_title("Logical error-rate reduction (moderate noise, $p \\in [0.02, 0.08]$)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_runtime(runtime_rows, path):
    """Grouped bar chart: per-trial loop vs batched pipeline wall-clock."""
    if not runtime_rows:
        return
    labels = [r["config"] for r in runtime_rows]
    loop = [r["loop_s"] for r in runtime_rows]
    batch = [r["batch_s"] for r in runtime_rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, loop, w, label="per-trial loop (batch=1)", color="#9aa0a6")
    ax.bar(x + w / 2, batch, w, label="batched pipeline", color="#1f77b4")
    for i in range(len(labels)):
        ax.text(i, max(loop[i], batch[i]), f"-{runtime_rows[i]['reduction_pct']:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("wall-clock seconds (10k trials)")
    ax.set_title("Batch simulation pipeline speed-up")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
