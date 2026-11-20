#!/usr/bin/env python3
"""verify_numbers.py -- recompute every derived quantity reported in the paper.

Each check pairs (a) the inputs, transcribed from Hu et al. (arXiv:2511.16665v2) or
stated as a modeling assumption, with (b) the value printed in the paper, and
asserts agreement to the printed precision. Run:

    python verify_numbers.py            # requires numpy only

Exit status is non-zero if any check fails.
"""
import math
import sys

import numpy as np

results = []


def check(name, computed, printed, tol, source):
    ok = abs(computed - printed) <= tol
    results.append((ok, name, computed, printed, source))


# ---------------------------------------------------------------- geometric means
gm = lambda v: math.prod(v) ** (1 / len(v))
check("TLT geomean, H100", gm([1.76, 1.71, 2.12, 1.86]), 1.86, 0.01, "TLT Fig. 11")
check("TLT-Base geomean, H100", gm([1.42, 1.38, 1.54, 1.31]), 1.41, 0.01, "TLT Fig. 11")
check("TLT geomean, A100", gm([1.73, 1.70, 2.07, 1.66]), 1.79, 0.01, "TLT Fig. 11")
check("TLT-Base geomean, A100", gm([1.38, 1.31, 1.49, 1.23]), 1.35, 0.01, "TLT Fig. 11")

# ---------------------------------------------------------------- ratios from tables
for gpu, w, wo, s in [("B200", 605.05, 259.71, 2.33), ("H100", 430.24, 164.65, 2.61),
                      ("A100", 259.01, 92.83, 2.79), ("RTX 5090", 293.84, 100.89, 2.91),
                      ("RTX 4090", 187.44, 65.28, 2.87), ("RTX 3090", 166.41, 51.75, 3.22)]:
    check(f"rollout SD speedup {gpu}", w / wo, s, 0.005, "TLT Table 2")
for m, t, s in [("Eagle", 542.12, 2.24), ("HASS", 553.91, 2.29), ("Eagle-3", 617.42, 2.55)]:
    check(f"SD method speedup {m}", t / 242.11, s, 0.005, "TLT Table 7")
check("async ckpt 893/280", 893 / 280, 3.2, 0.05, "TLT Fig. 17a")
check("selective async ckpt 893/97", 893 / 97, 9.2, 0.05, "TLT Fig. 17a")
check("selectivity share 280/97 (derived)", 280 / 97, 2.9, 0.05, "TLT Fig. 17a")
check("sequence packing 29.6/13.3", 29.6 / 13.3, 2.2, 0.05, "TLT Fig. 17b")
check("bucketed vs naive 30.39/10.69", 30.39 / 10.69, 2.8, 0.05, "TLT Table 5")
check("bucketed vs single 10.69/7.81 (derived)", 10.69 / 7.81, 1.37, 0.005, "TLT Table 5")
check("naive vs single 30.39/7.81 (derived)", 30.39 / 7.81, 3.89, 0.005, "TLT Table 5")
check("case study 337 s / 138 s", 337 / 138, 2.44, 0.005, "TLT Fig. 14")
check("accept-length gain, RL (%)", 100 * (6.53 - 4.59) / 4.59, 42.3, 0.05, "TLT Table 6")
check("accept-length gain, downstream (%)", 100 * (5.15 - 3.76) / 3.76, 37.0, 0.05, "TLT Table 6")

# ---------------------------------------------------------------- cost arithmetic
check("ByteDance trace GPU-hours", 128 * 11 * 24, 33792, 0, "TLT Sec. 1 (128 GPUs, 11 days)")
for S, pct in [(1.7, 41), (1.86, 46), (2.12, 53)]:
    check(f"step-time reduction at {S}x (%)", 100 * (1 - 1 / S), pct, 0.5, "derived: 1 - 1/S")
check("GPU-hour savings at 1.7x", 33792 * (1 - 1 / 1.7), 14000, 150, "derived (illustrative)")
check("GPU-hour savings at 1.86x", 33792 * (1 - 1 / 1.86), 16000, 400, "derived (illustrative)")

# ---------------------------------------------------------------- Pareto order statistic (Sec. IV)
a, B = 2.0, 128
exact = math.exp(math.lgamma(B + 1) + math.lgamma(1 - 1 / a) - math.lgamma(B + 1 - 1 / a))
median = 2 ** (1 / a)
rng = np.random.default_rng(20261119)
mc = np.mean([((1 - rng.random(B)) ** (-1 / a)).max() for _ in range(200_000)])
check("Pareto E[M_B]/x_min, exact", exact, 20.07, 0.005, "Eq. (2)")
check("Pareto E[M_B]/x_min, Monte Carlo (2e5, seed 20261119)", mc, 20.12, 0.005, "Eq. (2), simulation")
check("expected max / median", exact / median, 14.2, 0.05, "Sec. IV")
check("dashboard closed form sqrt(B)*a/(a-1)", B ** 0.5 * a / (a - 1), 22.63, 0.005, "companion model")
check("dashboard form / median", B ** 0.5 * a / (a - 1) / median, 16.0, 0.05, "companion model")
check("origin of the erroneous 5.7x: sqrt(B)/a", B ** 0.5 / a, 5.66, 0.005, "Sec. IV correction")

# ---------------------------------------------------------------- SD model (Sec. VI-A/B)
En = lambda al, g: (1 - al ** (g + 1)) / (1 - al)
check("E[n] limit alpha->1, gamma=8", En(1 - 1e-9, 8), 9.0, 1e-3, "Eq. (3)")
check("E[n] cap at alpha=0.8", 1 / (1 - 0.8), 5.0, 1e-9, "Eq. (3)")
for c, gstar in [(4, 3), (8, 5), (16, 7)]:
    g = np.arange(1, 200)
    check(f"optimal depth at c={c}, alpha=0.8", g[np.argmax(En(0.8, g) / (1 + g / c))], gstar, 0,
          "Eq. (4)")

# ---------------------------------------------------------------- threshold model (Sec. VI-C)
k = np.linspace(0, 200, 2_000_001)
check("argmax_k k*exp(-k/16)", k[np.argmax(k * np.exp(-k / 16))], 16.0, 1e-3, "Eq. (5)")

# ---------------------------------------------------------------- Amdahl (Sec. VI-E)
f = 0.85
s_of = lambda S: f / (1 / S - (1 - f))
check("Amdahl ceiling 1/(1-f)", 1 / (1 - f), 6.7, 0.05, "Eq. (7)")
check("implied rollout speedup at S=1.66", s_of(1.66), 1.9, 0.05, "Eq. (7)")
check("implied rollout speedup at S=2.12", s_of(2.12), 2.6, 0.05, "Eq. (7)")

# ---------------------------------------------------------------- report
width = max(len(r[1]) for r in results)
fails = 0
for ok, name, comp, printed, src in results:
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  computed={comp:<12.4f} printed={printed:<10} [{src}]")
print(f"\n{len(results) - fails}/{len(results)} checks passed")
sys.exit(1 if fails else 0)
