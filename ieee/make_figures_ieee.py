#!/usr/bin/env python3
"""
make_figures_ieee.py -- Figures for the IEEE version of
"Overlapping, Not Approximating" (E. Hibbs).

Every plotted value is an inline array so it can be audited line-by-line.
Data provenance:
  MEASURED  = transcribed from Hu et al., ASPLOS '26 (arXiv:2511.16665v2);
              figure/table number given per array. Re-checked against the
              arXiv v2 PDF on 2026-11-19.
  MODELED   = analytical, assumption-based (no measured data).
Run:  python make_figures_ieee.py      (matplotlib >= 3.8, numpy)
Outputs vector PDFs in ./figures/ sized for IEEE columns
(single column = 3.5 in, double column = 7.16 in).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
C = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
         blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#999999",
         black="#222222")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# --------------------------------------------------------------- Fig. 2
def fig_e2e():
    """MEASURED [TLT Fig. 11]. Normalized throughput, VeRL = 1.00.
    Open-R1 shows no bar for Llama-70B in either panel (NaN)."""
    models = ["Qwen-7B", "DeepSeek-7B", "Qwen-32B", "Llama-70B", "Geomean"]
    H100 = {"Open-R1": [0.25, 0.22, 0.07, np.nan, 0.18],
            "VeRL": [1, 1, 1, 1, 1],
            "TLT-Base": [1.42, 1.38, 1.54, 1.31, 1.41],
            "TLT": [1.76, 1.71, 2.12, 1.86, 1.86]}
    A100 = {"Open-R1": [0.24, 0.21, 0.06, np.nan, 0.17],
            "VeRL": [1, 1, 1, 1, 1],
            "TLT-Base": [1.38, 1.31, 1.49, 1.23, 1.35],
            "TLT": [1.73, 1.70, 2.07, 1.66, 1.79]}
    order = ["Open-R1", "VeRL", "TLT-Base", "TLT"]
    col = {"Open-R1": C["grey"], "VeRL": C["sky"], "TLT-Base": C["orange"],
           "TLT": C["green"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.9), sharey=True)
    x = np.arange(len(models)); w = 0.2
    for ax, (title, D) in zip(axes, [("NVIDIA H100", H100), ("NVIDIA A100", A100)]):
        for i, s in enumerate(order):
            v = np.array(D[s], dtype=float)
            ax.bar(x + (i - 1.5) * w, np.nan_to_num(v), w, label=s, color=col[s],
                   hatch=("////" if s == "Open-R1" else None),
                   edgecolor="white", linewidth=0.4)
            if s in ("TLT", "TLT-Base"):
                for xi, vi in zip(x + (i - 1.5) * w, v):
                    ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom",
                            fontsize=5.5, color=C["black"])
            if s == "Open-R1":
                for xi, vi in zip(x + (i - 1.5) * w, v):
                    if np.isnan(vi):
                        ax.text(xi, 0.04, "n/r", ha="center", fontsize=5, rotation=90)
        ax.axhline(1.0, color=C["black"], lw=0.6, ls="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_title(title); ax.set_ylim(0, 2.4)
    axes[0].set_ylabel("Normalized throughput\n(VeRL = 1.0)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_e2e")


# --------------------------------------------------------------- Fig. 3
def fig_batch():
    """MEASURED [TLT Table 4]. Qwen-32B, H100, TP=4, Draft_Depth=10, topK=8."""
    verify = [16, 32, 48, 64]
    BS = {1: [3.22, 3.46, 3.56, 3.62], 2: [3.08, 3.28, 3.39, 3.38],
          4: [3.01, 3.09, 3.13, 2.98], 8: [2.73, 2.63, 2.51, 2.27],
          16: [2.67, 2.52, 2.24, 1.91], 32: [2.48, 2.23, 1.90, 1.70]}
    cmap = [C["blue"], C["sky"], C["green"], C["purple"], C["orange"], C["verm"]]
    mk = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for (bs, v), c, m in zip(BS.items(), cmap, mk):
        ax.plot(verify, v, "-", marker=m, color=c, label=f"BS={bs}", lw=1.2, ms=3.5)
    ax.set_xlabel("Tokens to verify per step"); ax.set_ylabel("SD speedup (×)")
    ax.set_xticks(verify); ax.set_ylim(1.0, 3.8)
    ax.legend(ncol=3, loc="lower left")
    save(fig, "fig_batch")


# --------------------------------------------------------------- Fig. 4
def fig_model():
    """MODELED. (a) speedup(gamma) = E[n]/(1+gamma/c), alpha=0.8 (uniform accept);
    (b) aggregate benefit k*b(k) = b0*k*exp(-k/tau), b0=1, tau=16."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.75))
    g = np.arange(1, 25); al = 0.8
    En = (1 - al ** (g + 1)) / (1 - al)
    for c, col, m in [(4, C["verm"], "o"), (8, C["blue"], "s"), (16, C["green"], "^")]:
        sp = En / (1 + g / c); i = int(np.argmax(sp))
        a1.plot(g, sp, "-", color=col, lw=1.1, label=f"c={c}")
        a1.plot(g[i], sp[i], m, color=col, ms=3.5)
    a1.set_xlabel(r"draft depth $\gamma$"); a1.set_ylabel("modeled speedup (×)")
    a1.set_title(r"(a) $\alpha=0.8$; optima $\gamma^*=3,5,7$")
    a1.set_ylim(0.8, 3.6)
    a1.legend(loc="upper right", handlelength=1.0, ncol=3, columnspacing=0.5,
              fontsize=6)
    k = np.linspace(0, 80, 400); tau = 16
    a2.plot(k, k * np.exp(-k / tau), color=C["blue"], lw=1.2)
    a2.axvline(tau, color=C["verm"], ls="--", lw=0.8)
    a2.text(tau + 3, 1.0, r"$k=\tau$", color=C["verm"], fontsize=7)
    a2.set_xlabel(r"active requests $k$"); a2.set_ylabel(r"$k\,b(k)$ (a.u.)")
    a2.set_title(r"(b) $\tau=16$ (fitted)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_model")


# --------------------------------------------------------------- Fig. 5
def fig_amdahl():
    """Amdahl bound with rollout fraction f = 0.85 [TLT Sec. 1, Fig. 1a].
    Band (x): BS=1 rollout SD speedups 2.33-3.22x [TLT Table 2].
    Band (y): end-to-end TLT speedups 1.66-2.12x [TLT Fig. 11]."""
    f = 0.85
    s = np.linspace(1, 8, 400)
    S = 1 / ((1 - f) + f / s)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axvspan(2.33, 3.22, color=C["sky"], alpha=0.25, lw=0,
               label="Rollout SD speedup, BS=1 (meas.)")
    ax.axhspan(1.66, 2.12, color=C["orange"], alpha=0.25, lw=0,
               label="End-to-end TLT speedup (meas.)")
    ax.plot(s, S, color=C["black"], lw=1.2, label=r"Amdahl bound, $f=0.85$")
    ax.axhline(1 / (1 - f), color=C["grey"], ls=":", lw=0.8)
    ax.text(5.3, 1 / (1 - f) + 0.15, r"limit $1/(1-f)=6.67\times$", fontsize=6.5,
            color=C["grey"])
    for e2e in (1.66, 2.12):
        si = f / (1 / e2e - (1 - f))
        ax.plot(si, e2e, "o", color=C["verm"], ms=3)
    ax.set_xlabel(r"rollout-phase speedup $s$ (×)")
    ax.set_ylabel("end-to-end speedup (×)")
    ax.set_xlim(1, 8); ax.set_ylim(1, 7.4)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90))
    save(fig, "fig_amdahl")


if __name__ == "__main__":
    for fn in (fig_e2e, fig_batch, fig_model, fig_amdahl):
        fn()
