# Provenance Self-Audit

Self-check of `paper.md` against the accuracy rules, plus the result of an independent multi-agent audit. Four parts: (A) direct-quote audit; (B) dashboard-equation audit; (C) discrepancies found between paper and dashboard + fixes; (D) independent adversarial audit result.

---

## A. Direct-quote audit

Every double-quoted prose string in the **final** `paper.md`, its source, word count, and confirmation it is verbatim and < 15 words. (Numbers shown in quotes — "1.7–2.1×", "5.7×", "+42%" — are figures, not prose quotations, listed for completeness.)

| Quote (verbatim) | Source | Words | < 15? | Verbatim? |
|---|---|---:|:--:|:--:|
| "over 1.7×" | Paper, Abstract | 2 | ✅ | ✅ |
| "1.7–2.1×" (figure) | Paper, Fig 11 caption | — | ✅ | ✅ (caption prints "1.7-2.1×") |
| "in a non-blocking manner" | Paper, §4.2 (single use, in §4.1) | 5 | ✅ | ✅ |
| "Mathematically Lossless" | Paper, §2.2 | 2 | ✅ | ✅ |
| "memory-bound towards compute-bound" | Paper, §2.2 | 3 | ✅ | ✅ |
| "Lossless Guarantee" | Paper, §3 (verified vs PDF; data_table row 79) | 2 | ✅ | ✅ |
| "Interference-Free" | Paper, §3 (verified vs PDF; data_table row 79) | 1 | ✅ | ✅ |
| "free byproduct" | Paper, Abstract (corrected from §3 — see §C-4) | 2 | ✅ | ✅ |
| "spot tasks" | Paper, §4.2 | 2 | ✅ | ✅ |
| "over 20×" | Paper, §4.1 | 2 | ✅ | ✅ |
| "under-utilized zone" | Paper, Fig 2 (annotation "Under-Utilized Zone") | 2 | ✅ | ✅ (case-normalized) |
| "a few very long responses dominate execution time" | Paper, Abstract | 8 | ✅ | ✅ |
| "training-time test" | Paper, §6.5 / Fig 7 | 2 | ✅ | ✅ |
| "mixes multiple policy versions" | Paper, §7 | 4 | ✅ | ✅ |
| "without introducing additional loss or policy staleness" | Paper, §7 | 7 | ✅ | ✅ |
| "≈ 5.7×" (figure) | Dashboard, §1 | — | ✅ | ✅ (reported as the dashboard's; see §C-1) |
| "+42%" (figure) | Dashboard, §4 | — | ✅ | ✅ |
| "drops to 1.08× at BS=32" | Dashboard, §3 | 5 | ✅ | ✅ (reported as the dashboard's error; flagged, §C-2) |

**Result:** every prose quote is < 15 words (longest = 8) and verbatim against the source text. No quote ≥ 15 words. One short quote per source per claim; all other uses of each source are paraphrased.

**Quote-integrity fixes applied after the independent audit (§D):**
- **Removed** the quotation "∼85% of the total step time" — the paper's contiguous string is "(∼85%) of the total step time", so the parens make my version non-contiguous; demoted to paraphrase ("approximately 85% of total step time").
- **De-duplicated** "in a non-blocking manner" — it appeared twice; kept once (§4.1, verbatim source §4.2), paraphrased the §1.2 use.
- **Verified** "Lossless Guarantee" and "Interference-Free" are verbatim §3 design-principle names; added `data_table.md` row 79 as their audit-trail entry.
- **Restructured** the §5.3 dashboard-error sentence and **removed** the non-contiguous quotation "3.66× at BS=1" (now paraphrase); kept "drops to 1.08× at BS=32" as a flagged verbatim dashboard quote.

*Repeated-source note:* the Abstract is quoted for three **distinct** claims ("over 1.7×"; "a few very long responses dominate execution time"; "free byproduct") — one per claim; all other Abstract references are paraphrased.

---

## B. Dashboard-equation audit

Every "Model Math" derivation used in `paper.md`, with confirmation it is (i) tagged `[Dashboard §N]`, (ii) assumption-flagged, and (iii) **not** presented as a paper result.

| Dashboard item | Used in | `[Dashboard]`? | Assumption flagged? | Reads as paper-validated? |
|---|---|:--:|:--:|:--:|
| $T_{\text{batch}}=\max_i L_i/\text{thr}_i$ (§1) | §3.1 | ✅ | n/a (definition) | ❌ no |
| Pareto CDF $F(x)=1-(x_{\min}/x)^\alpha$ (§1) | §3.2 | ✅ | ✅ Pareto vs empirical Fig 1a | ❌ no |
| $\mathbb{E}[M_B]=x_{\min}B^{1/\alpha}\frac{\alpha}{\alpha-1}$ (§1) | §3.2 | ✅ | ✅ | ❌ no |
| $\mathbb{E}[n]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ (§2) | §5.1 | ✅ | ✅ uniform $\alpha$ | ❌ no |
| $\text{speedup}=\mathbb{E}[n]/(1+\gamma/c)$ (§2) | §5.2 | ✅ | ✅ (inherits §2) | ❌ no |
| $b(k)=b_0 e^{-k/\tau}$, $k\,b(k)\ge O_{\text{SD}}$ (§3) | §5.3 | ✅ | ✅ $\tau=16$ not in paper | ❌ no |
| $\mathcal{L}=\mathbb{E}_{x\sim\pi_t}[-\log p_\theta]$ (§4) | §5.4 | ✅ | ✅ (inherits §4) | ❌ no |
| KL recursion incl. $+O(\eta_d^2)$ (§4) | §5.4 | ✅ | ✅ Fisher ≈ const | ❌ no |
| Condition $\eta_d I_t>\delta_\pi$ (§4) | §5.4, §7.2 | ✅ | ✅ | ❌ no |
| Variant A $k^*(t)=k_0(1+\lambda V(t)/V(0))$ (§5) | §8 | ✅ | ✅ not evaluated | ❌ no |
| Variant B $\mathbb{E}[n_{\text{ens}}]=\sum_j w_j\mathbb{E}[n_j]$ (§5) | §8 | ✅ | ✅ | ❌ no |
| Variant C $\sigma^*=\mathrm{argsort}_i(-\hat L_i/c_i)$ (§5) | §8 | ✅ | ✅ | ❌ no |
| Open problems 1–4 (§6) | §8 | ✅ | ✅ | ❌ no |
| "+42%" framing on Tbl 6 (§4) | §5.4 | ✅ + arithmetic [Author synthesis] | ✅ | ❌ no |
| Fig 16 read as ~32% vs ~6% (§4) | §5.4 | ✅ (dashboard's reading) | ✅ | ❌ no |

**Result:** all 15 dashboard items tagged `[Dashboard]`, each assumption-flagged, none presented as paper-validated. My own derivations (geometric-series steps, exact Pareto order statistic with $\Gamma(1-1/\alpha)$, the formula readings) are tagged `[Author synthesis]` and kept distinct.

---

## C. Discrepancies found between paper and dashboard (and fixes)

1. **Dashboard "≈5.7× vs median" (§1) is internally inconsistent.** Its own form gives $\mathbb{E}[M_B]\approx22.6\,x_{\min}$ at $\alpha=2,B=128$; the Pareto median is $x_{\min}\sqrt2$, so the ratio is $\approx16\times$, not $5.7\times$. **Handling:** §3.2 reports it strictly as the dashboard's figure, with the inconsistency flagged `[Author synthesis]`; not adopted.

2. **Dashboard "1.08× at BS=32 / 3.66× at BS=1, per Table 4" (§3) contradicts the paper.** Tbl 4 shows BS=32 from 1.70× (verify-64) to 2.48× (verify-16); no 1.08×; 3.66× is the Fig 13 peak. **Handling:** §5.3 uses the dashboard's threshold *model* but anchors all batch numbers to Tbl 4 and explicitly flags the dashboard's figures as wrong. Not propagated.

3. **Dashboard prefactor vs exact order statistic.** Exact tail constant is $\Gamma(1-1/\alpha)$ ($\approx1.77$ at $\alpha=2$); dashboard substitutes the Pareto mean $\frac{\alpha}{\alpha-1}=2$. **Handling:** §3.2 derives the exact form `[Author synthesis]`, identifies the substitution, notes it mildly over-states.

4. **Quote-attribution fix.** "free byproduct" (unhyphenated) is verbatim from the **Abstract**; §3 uses "by-product". Citation corrected `[Paper §3]`→`[Paper Abstract]`.

5. **OCR-limited values, not over-claimed.** Fig 11 Open-R1 per-bar values (range only) and the Fig 13 heatmap interior (only the two extrema) — recorded in `data_table.md` "non-uses".

6. **Reported-both, per the rules.** Headline given as "over 1.7×" (Abstract) **and** "1.7–2.1×" (Fig 11), each with context.

---

## D. Independent multi-agent audit (workflow `tlt-provenance-audit`, 7 agents)

Six adversarial auditors checked the finished `paper.md` against verified ground truth, each on one dimension. Outcome:

| Dimension | Verdict | Findings |
|---|:--:|---|
| Paper-number fidelity | ✅ PASS | 0 — "40+ distinct numerical claims … every claim exactly accurate, correct values, units, and conditions" |
| Dashboard attribution | ✅ PASS | 0 — all `[Dashboard]` items assumption-flagged; both known inconsistencies surfaced; nothing reads as paper-validated |
| Math correctness | ✅ PASS | 0 — geometric series, Pareto order statistic, KL recursion w/ $O(\eta_d^2)$, $\eta_d I_t>\delta_\pi$, $w_j=|B_j|/B$ all verified |
| Structure/completeness | ✅ PASS | 3 minor, all self-resolving (no fix) — all 12 section categories present; 78 `§` cross-refs valid |
| Cross-file consistency | ✅ PASS | 0 — every number matches across `paper.md`/`data_table.md`; tags consistent; +42% arithmetic confirmed |
| Quote integrity | ⚠️ → ✅ | 1 critical + 3 major — **all four now fixed** (see §A) |

**Resolution of the four quote findings:**
- **C1** (85% quote not strictly contiguous) → demoted to paraphrase. ✅
- **M1** ("in a non-blocking manner" used twice) → de-duplicated to one verbatim use (§4.1). ✅
- **M2** ("Lossless Guarantee" lacked an audit entry) → confirmed verbatim in §3, added `data_table.md` row 79. ✅
- **M3** (nested dashboard-error quotes) → restructured; non-contiguous "3.66× at BS=1" demoted to paraphrase; "drops to 1.08× at BS=32" kept as a flagged verbatim dashboard quote. ✅

The audit found **no wrong number, no math error, no misattribution, and no fabrication** — the only defects were quotation-integrity issues, now resolved.

---

## Verdict

- All direct quotes: verbatim, < 15 words, sourced, one-per-source-per-claim. ✅
- All dashboard derivations: tagged secondary + assumption-flagged + not presented as paper results. ✅
- Paper↔dashboard discrepancies: identified, isolated, not propagated; contested dashboard figures appear only *as the dashboard's*, explicitly flagged. ✅
- No fabricated values or citations; every external reference is one the paper actually cites. ✅
- Independent 6-dimension audit: 5/6 clean on first pass; quotes dimension cleared after four fixes. ✅

**Status: provenance-clean.**
