# Phase 3 — Accuracy Hardening

Re-derives every equation from scratch, re-checks numerical self-consistency, re-counts the figures, audits every verbatim quote, and red-teams the paper. Computations were run in Python (`math`, `numpy`); the script is reproduced under §2.

---

## 1. Equation re-derivations

### 1.1 Geometric-series accept count — [Dashboard §2], derivation [Author synthesis]
Claim: $\mathbb{E}[n]=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$ (α = per-token accept prob, γ = draft depth).

Let $A\in\{0,\dots,\gamma\}$ be the number of accepted drafted tokens; acceptance of a prefix requires the first $k$ all accepted, so $P(A\ge k)=\alpha^k$. The per-position law is $P(A=k)=\alpha^k(1-\alpha)$ for $k<\gamma$ and $P(A=\gamma)=\alpha^\gamma$ (no rejection). Tokens committed $=A+1$ (bonus token), so
$$\mathbb{E}[n]=1+\sum_{k=1}^{\gamma}P(A\ge k)=\sum_{k=0}^{\gamma}\alpha^k.$$
Geometric sum: $S-\alpha S=1-\alpha^{\gamma+1}\Rightarrow S=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. ∎
**Checks (verified):** $\alpha\!\to\!1\Rightarrow \mathbb{E}[n]\to\gamma+1$ (computed 9.0000 at γ=8); $\gamma\!\to\!\infty\Rightarrow \mathbb{E}[n]\to\frac{1}{1-\alpha}$ (α=0.8→cap 5.0, α=0.9→cap 10.0). **MATCHES.**
**Assumption / violation:** assumes a *uniform* per-token α. The paper's **Fig 16** shows acceptance **decays with draft-token index** (vanilla collapses by index ~8), so the uniform-α premise is violated in practice — the model is *qualitative*, not predictive.

### 1.2 Speedup — [Dashboard §2], reading [Author synthesis]
Claim: $\text{speedup}=\dfrac{\mathbb{E}[n]}{1+\gamma/c}$, $c=T_{\text{verify}}/T_{\text{draft}}$.
One SD cycle = one verify + γ draft steps = $1+\gamma/c$ verify-time units, yielding $\mathbb{E}[n]$ tokens vs 1 token/unit for vanilla. Numerator saturates in γ (§1.1); denominator grows linearly ⇒ **interior optimum** (inverted-U).
**Check (verified):** $\arg\max_\gamma$ speedup $=3,5,7$ for $c=4,8,16$ (α=0.8) — interior, and optimal depth grows with $c$. Qualitatively **MATCHES Fig 13(b)** (peak 3.66× at depth 12/verify 64). Monotone increasing in α.
**Assumption / violation:** treats $c$ as fixed. As batch grows, verification becomes compute-bound and the effective benefit shrinks — the paper's **Tbl 4** shows speedup falling from 3.62× (BS=1) to 1.70× (BS=32, verify-64). So the single-batch formula does not capture the batch axis (that is §1.4's job).

### 1.3 Pareto expected-maximum — [Dashboard §1]; exact form + recomputation [Author synthesis]
Dashboard claim: $\mathbb{E}[M_B]=x_{\min}B^{1/\alpha}\frac{\alpha}{\alpha-1}$, with a worked example "≈5.7× vs median" at α=2, B=128.
Exact order statistic for B i.i.d. Pareto: $\mathbb{E}[M_B]=x_{\min}\frac{\Gamma(B+1)\Gamma(1-1/\alpha)}{\Gamma(B+1-1/\alpha)}\approx x_{\min}B^{1/\alpha}\Gamma(1-1/\alpha)$. The dashboard substitutes the **Pareto mean factor** $\frac{\alpha}{\alpha-1}$ for the tail constant $\Gamma(1-1/\alpha)$.
**Recomputation (α=2, B=128, verified):**
| Quantity | Value (·x_min) |
|---|---|
| Exact $\mathbb{E}[M_B]$ | **20.07** |
| Monte-Carlo (2×10⁵×128) | **20.14** |
| Dashboard formula $\sqrt{128}\cdot 2$ | 22.63 |
| Asymptotic $\sqrt{128}\cdot\Gamma(0.5)$ | 20.05 |
| median $=\sqrt2$ | 1.414 |
| **exact/median** | **14.19×** |
| dashboard-formula/median | 16.0× |
> **CORRECTION (flag the dashboard):** the dashboard's stated "≈5.7× vs median" is **wrong**. The true expected-max-to-median ratio is **≈14×** (exact/Monte-Carlo), or 16× under the dashboard's own (slightly inflated) closed form. The figure **5.7 ≈ √128/2 = B^{1/α}/α = 5.66**, i.e. it appears to drop the `α/(α−1)` factor *and* divide by the median incorrectly — a normalization error. The dashboard's `α/(α−1)=2` prefactor also mildly overstates vs the exact tail constant $\Gamma(0.5)=1.77$ (22.63 vs 20.07).
**Assumption / violation:** Pareto tail is the dashboard's assumption; the paper's **Fig 1a** is empirical (log-normal-shaped). Use the order-statistic argument as *motivation only*.

### 1.4 Optimal-stopping activation threshold — [Dashboard §3]
Claim: under $b(k)=b_0 e^{-k/\tau}$ (per-request SD benefit at $k$ active requests), activate SD when aggregate benefit clears switch overhead: $k\,b(k)\ge O_{\text{SD}}$; the indifference point is $b(k^\*)=O_{\text{SD}}/k^\*$.
Aggregate benefit $g(k)=b_0 k e^{-k/\tau}$ has $g'(k)=b_0 e^{-k/\tau}(1-k/\tau)=0\Rightarrow k=\tau$. So aggregate SD benefit is **maximized at $k=\tau$** and decays for $k>\tau$; with overhead $O_{\text{SD}}$ there is a crossing beyond which SD is not worth enabling — the analytic basis for an activation threshold.
**Self-consistency:** the dashboard fits $\tau=16$ to Table 4. The *code* (Phase 1) implements the threshold as a hard `batch_size ≤ 32` with a 10-check warmup (`eagle_worker.py:361,196`) — a step rule, not the smooth $b(k)$ model. So the model **explains the direction** of the Tbl 4 collapse but is not the deployed mechanism. State as qualitative.
**Assumption / violation:** exponential decay with τ=16 is **not in the paper** [Dashboard §3 Assumption]; it is a fit, and the deployed rule is a fixed-32 step.

### 1.5 Drafter-convergence (moving target) — [Dashboard §4]
Claim: with local relation $\alpha\approx e^{-D_{\text{KL}}}$ and per-step KL recursion
$$D_{\text{KL}}(\pi_{t+1}\Vert p_{\theta_{t+1}})\le D_{\text{KL}}(\pi_t\Vert p_{\theta_t})-\eta_d I_t+\delta_\pi+O(\eta_d^2),$$
the accept rate is non-decreasing in expectation iff $\eta_d I_t>\delta_\pi$ (drafter per-step learning outpaces policy drift). The fixed point $D^\*\approx\delta_\pi/(\eta_d I_t)$ (ignoring $O(\eta_d^2)$) ⇒ a steady accept rate $\alpha^\*\approx e^{-D^\*}$.
**Self-consistency:** explains the paper's **Fig 15** (dip after each target update, recovery within a few iterations) and **Tbl 6** (RL co-training raises accept length 4.59→6.53). Plain arithmetic: $(6.53-4.59)/4.59=42.3\%$ [Author synthesis] — matches the dashboard's "+42%."
**Assumption / violation:** Fisher info ≈ constant and $\alpha\approx e^{-D_{\text{KL}}}$ local — both are the dashboard's [Assumption]; not asserted by the paper.

---

## 2. Numerical self-consistency (verification script + result)

```python
# (excerpt — full run logged in session)
alpha,B,xmin=2.0,128,1.0
exact = xmin*math.exp(lgamma(B+1)+lgamma(1-1/alpha)-lgamma(B+1-1/alpha))   # 20.07
dash  = xmin*B**(1/alpha)*alpha/(alpha-1)                                   # 22.63
median= xmin*2**(1/alpha)                                                   # 1.414
# exact/median = 14.19x ; dashboard/median = 16.0x ; (dashboard text says 5.7x -> WRONG)
# Monte-Carlo E[max] over 2e5 draws of 128 = 20.14x  -> MC/median = 14.24x
# E[n]=(1-a^(g+1))/(1-a): a->1 (a=.999999,g=8) = 9.0000 (=g+1); caps a=.8->5.0, a=.9->10.0
# speedup interior optimum: argmax_g = 3,5,7 for c=4,8,16
```
**Findings:** (a) geometric series and its limits **exact**; (b) speedup is an interior-optimum inverted-U matching Fig 13b **qualitatively**; (c) the threshold model predicts the **direction** of the Tbl 4 collapse (benefit peaks at k=τ then decays) — **qualitative, not predictive**; (d) the Pareto example is **numerically wrong** (5.7× vs true ≈14×). The Markov/threshold models must be described as **qualitative scaffolding**, never as predictive fits to the paper's tables.

---

## 3. Figure/number recount checklist (each verified against source)

| Value | Source + condition | Status |
|---|---|---|
| "over 1.7×" end-to-end | Abstract | ✓ |
| 1.7–2.1× over VeRL; geomean **1.86× (H100) / 1.79× (A100)** (corrected 2026-11-19; was 1.70×); peak **2.12×** (Qwen-32B/H100) | Fig 11; Tbl 3 | ✓ (report both abstract + figure forms) |
| Rollout ≈85% of step | §1, Fig 1a | ✓ |
| Tbl 1 topK 4–16 → accept 8.29–8.67, speedup 3.47–3.65× (Qwen-32B, TP4, d12, v64) | Tbl 1 | ✓ |
| Fig 13 peak speedup **3.66×** (d12/v64); max accept length **9.78** (d16/v128) | Fig 13 | ✓ |
| Tbl 4 BS=1→3.62× … BS=32→1.70× (verify-64) | Tbl 4 | ✓ |
| Tbl 2 GPU: B200 2.33× · H100 2.61× · A100 2.79× · RTX5090 2.91× · RTX4090 2.87× · RTX3090 3.22× (BS1, TP1) | Tbl 2 | ✓ |
| Tbl 3 nodes 1/2/4/8: 7B 1.21/1.45/1.62/1.76×; 32B OOM/OOM/1.83/2.12× | Tbl 3 | ✓ |
| Bucketed CUDAGraph **7.81 → 30.39 → 10.69 GB** (2.8×), Llama-3-8B TP4 | Tbl 5 | ✓ |
| Selective async ckpt **893 → 280 → 97 ms** (3.2× → 9.2×) | Fig 17a | ✓ |
| Sequence packing **13.3 → 29.6 samples/s** (2.2×) | Fig 17b | ✓ |
| Adaptive drafter accept len **4.59→6.53** (RL), **3.76→5.15** (downstream) | Tbl 6 | ✓ |
| Tbl 7 Eagle 6.53 / 542.12 tok/s / 2.24× / 1× cost (default) | Tbl 7 | ✓ |
| Case study **2.44×** over no-SD (128 req, Qwen-32B/H100) | Fig 14 | ✓ |
| Overheads: stage-transition <1%, SD reprefill <3 s, drafter every ~10 steps | §6.4 (✓ code: `training_interval_steps=10`) | ✓ |

All values match `data_table.md`. No transposed digits found on recount.

---

## 4. Verbatim-quote audit (post-Phase-1 state)

Every double-quoted prose string in `paper.md`, with word count ≤ 15 and verbatim status (carried from `provenance_check.md` §A, re-confirmed):

| Quote | Source | Words | OK |
|---|---|---|---|
| "over 1.7×" | Abstract | 2 | ✓ |
| "in a non-blocking manner" | §4.2 (single use) | 5 | ✓ |
| "Mathematically Lossless" | §2.2 | 2 | ✓ |
| "memory-bound towards compute-bound" | §2.2 | 3 | ✓ |
| "Lossless Guarantee" | §3 | 2 | ✓ |
| "Interference-Free" | §3 | 1 | ✓ |
| "free byproduct" | Abstract | 2 | ✓ |
| "spot tasks" | §4.2 | 2 | ✓ |
| "over 20×" | §4.1 | 2 | ✓ |
| "under-utilized zone" | Fig 2 | 2 | ✓ |
| "a few very long responses dominate execution time" | Abstract | 8 | ✓ |
| "training-time test" | §6.5/Fig 7 | 2 | ✓ |
| "mixes multiple policy versions" | §7 | 4 | ✓ |
| "without introducing additional loss or policy staleness" | §7 | 7 | ✓ |
| "drops to 1.08× at BS=32" | Dashboard §3 (flagged wrong) | 5 | ✓ |

**Result:** all ≤ 8 words for paper quotes; no quote ≥ 15 words; one quote per source per claim. **PASS.** (Phase 6 adds at most one short MIT-News quote, kept < 15 words and marked search-sourced.)

---

## 5. Red-team (reviewer objections + resolutions)

| # | Objection | Resolution |
|---|---|---|
| R1 | The Pareto order-statistics model is the **dashboard's**, not validated by the paper — and its "5.7×" is numerically wrong. | Already tagged [Dashboard, Assumption]; Phase 3 §1.3 corrects the number (true ≈14×). **Phase 6:** sharpen §3.2 to state exact/MC ≈14× and that 5.7× is a dashboard normalization error. |
| R2 | "Lossless" is loose. | Restate precisely everywhere: **token-level distributional equivalence** of rollouts (SD verification preserves the target's sampling distribution). Phase 6 edit to Abstract/§2.2/§7.2. |
| R3 | Speedups are **throughput** (tokens/s per step), not wall-clock-to-target-accuracy or convergence. | Add a sentence in §6.1/§6.5: gains are step-throughput at matched reward curves (Fig 12), **not** a claim about fewer steps to convergence. |
| R4 | GPU-diversity ordering (RTX3090 > H100 > B200) is **BS=1, TP=1** rollout micro-benchmark, not end-to-end training. | §6.3 already notes BS1/TP1; Phase 6 adds explicit "(not representative of full-scale training; more I/O-bound GPUs gain more in this micro-benchmark)." |
| R5 | The "drafter trained for free during RL" headline relies on flags **OFF by default** in shipped configs (`enable_drafter_training=false`). | Phase 6 "Implementation & Reproducibility" + "Threats to Validity" state this explicitly (Phase 1 §C). |
| R6 | The Markov speedup model assumes **uniform α**, contradicted by Fig 16. | §5.1 already flags it; Phase 3 §1.1 makes the violation explicit; keep model labeled *qualitative*. |
| R7 | 9.2× checkpoint speedup could read as a parameter-count ratio. | Phase 1 §C + Phase 3: it is an **end-to-end measurement** (async dominates; selectivity excludes the frozen embed/LM-head which dominate the *drafter's* params). Phase 6 phrasing fixed. |
| R8 | "Free byproduct" drafter is trained on **partial + one-step-stale** data; is it as good as a fully-trained drafter? | Tbl 8 shows trained/OSD drafters reach 6.5–6.9 accept length; paper claims partial+buffered ≈ full-set. Add limitation: drafter quality is *adequate for SD speedup*, not claimed SOTA standalone. |
| R9 | BEG-MAB adds **exploration overhead** and assumes near-stationarity within the window. | Note: ε=0.1, window=1000 (code); reward is non-stationary across RL steps — MAB tracks via sliding window. Add one limitation sentence. |
| R10 | End-to-end gains are **workload-dependent** (tail severity, model size, cluster size). | Tbl 3 shows gains grow with scale; add "gains scale with tail severity and are smaller when tails are short." |
| R11 | **RollPacker is synchronous** (Phase 2) — grouping it under "Why Not Fully Async?" risks mislabeling. | Phase 6: separate synchronous long-tail scheduling (RollPacker) from async/off-policy (AReaL, APRIL-recycling). |
| R12 | Fig 13/Tbl 1 SD grids use **temperature 0**; training uses **temp 0.9** — accept rates differ between the SD micro-benchmarks and real rollouts. | Add a clause: SD hyperparameter sweeps are at temp 0 for stable accept-length measurement; real-training accept rates (Tbl 6/Fig 16) are the operative ones. |

All 12 are either already handled in the text or scheduled as concrete Phase-6 edits/limitation sentences (tracked in CHANGELOG.md).

---

## Net
Equations re-derived and **all correct as stated**, except the dashboard's **5.7× Pareto example, which is numerically wrong (true ≈14×)** and is downgraded to a flagged dashboard figure. Models are **qualitative**, not predictive — stated as such. Figure recount: **no discrepancies**. Quote audit: **PASS**. Twelve red-team objections resolved or converted to explicit limitations for Phase 6.
