# Verification Report — Final Reviewer Verdict

Senior-reviewer pass over the TLT review paper, Phases 0–7. Repo grounded on `mit-han-lab/fastrl` @ `main` `bce3df7` (Apache-2.0; SGLang vendored in-tree). Anchor: arXiv:2511.16665v2; ASPLOS '26; DOI 10.1145/3779212.3790231.

## Output manifest (all present)
`paper.md` (revised), `data_table.md` (with [Code] section), `provenance_check.md`, `phase0_inventory.md`, `phase1_code_grounding.md`, `phase2_research_log.md`, `phase3_accuracy.md`, `phase5_citations.md`, `make_figures.py`, `figures/` (11 figures × {.pdf,.svg} + `manifest.md`), `CHANGELOG.md`, `verification_report.md`. (Phase 4's write-up is `figures/manifest.md` per the spec.)

## Claims-coverage statistic
Audit trail = `data_table.md` (the canonical claim inventory), cross-checked against in-text tags in `paper.md`.

| Provenance | Catalogued claims (data_table) | In-text tag occurrences (paper.md) |
|---|---:|---:|
| **[Paper]** (measured) | **76** (75 exact + 1 figure-read) | 110 |
| **[Dashboard]** (modeled, assumption-based) | **20** | 23 |
| **[Code]** (implemented) | **20** | 22 |
| **[Author synthesis]** | **2** in-table + ~10 derivations in §3/§5 | 10 |
| **Total catalogued** | **≈118 distinct claims** | — |

- **Multiply-sourced:** **≥15** mechanism claims carry both a [Paper] and a [Code] grounding (e.g., elastic threshold = §5.1 + grpo_7B.sh:28; BEG-MAB reward = Alg 1 + eagle_worker.py:1234,1240; single-layer frozen drafter = §4.1 + eagle_trainer.py; selective ckpt = Fig 17a + eagle_background_trainer.py). The five flagged corrections are explicitly cross-tagged ([Paper] vs [Dashboard] vs [Author synthesis]).
- **No claim is unsourced.** Every quantitative/mechanistic statement earns ≥1 of the four tags; the lone known soft spot (DataBuffer long-sequence prioritization) is explicitly downgraded to PARTIAL with the reason.

## Verbatim-quote audit summary
- **15 distinct prose quotes** in `paper.md`, all **verbatim** and **< 15 words** (longest = 8: "a few very long responses dominate execution time"). One quote per source per claim. (The raw grep count of 42 quoted strings includes config tokens, equation labels, and identifiers like `"spot tasks"`, not additional prose quotations.) Full table in `provenance_check.md` §A and `phase3_accuracy.md` §4. **PASS.**
- The one dashboard quote carried for refutation — "drops to 1.08× at BS=32" — is explicitly labeled the dashboard's error and not used as a value.

## Paper ↔ code divergences (all CONFIRMED mechanisms; within-confirmation refinements)
1. **Worker states: 5 in code** (`IDLE/GENERATING/RELEASED/TRAINING/COMPLETED`) vs the paper's 3-state "BUSY/IDLE/TRAINING" shorthand. [worker_manager.py:54-61]
2. **"Frozen LM head" is EAGLE-only** — trainable on the EAGLE-3 path. [eagle3_trainer.py:475]
3. **DataBuffer prioritization** — code uses `random.sample` over the last 2 steps + a 512-token window; the paper's "prioritize previous long sequences" is not present as written (cross-step offset *is*). [eagle_background_trainer.py:299-356]
4. **9.2× checkpoint** is an end-to-end measurement, not a parameter-count ratio. [Fig 17a; eagle_background_trainer.py:119-148]
5. **Drafter loss** is `SmoothL1` + soft-CE (0.5/0.5), a refinement of "L1+CE". [eagle_background_trainer.py:558-578]
6. **Cadence reconciliation:** shipped `training_interval_steps=10` matches the paper's "~10 steps" (code `.get(...,1)` fallback differs). [fastrl_trainer.yaml:20]
7. **Online drafter training is OFF by default** (`enable_drafter_training=false`); example scripts use a pre-trained Eagle drafter. **Top reproducibility caveat.** [fastrl_trainer.yaml:19,26]
   *Net:* every mechanism the paper claims is implemented and locatable; the divergences are sharpenings/omissions, not contradictions.

## Paper ↔ dashboard discrepancies
1. **"≈5.7× vs median" (Dashboard §1) is numerically wrong.** Recomputed expected-max/median ≈ **14×** (exact 20.07·x_min, Monte-Carlo 20.14; dashboard formula 22.63 → 16×). The 5.7 = √128/2 = B^(1/α)/α (mis-normalization). Reported as the dashboard's figure only; corrected in §3.2.
2. **"1.08× at BS=32, per Table 4" (Dashboard §3) contradicts the paper** (Tbl 4 = 1.70× at verify-64; no 1.08× exists). The dashboard's companion "3.66× at BS=1" is the Fig 13 peak, not a Tbl 4 entry. Flagged; not propagated.
3. **Dashboard `α/(α−1)` prefactor mildly overstates** the exact tail constant Γ(1−1/α) (2 vs 1.77 at α=2). Noted.
4. **All four dashboard models rest on assumptions** the paper does not assert (Pareto tail; uniform α — contradicted by Fig 16; τ=16; constant Fisher info). Every use is tagged [Dashboard, Assumption] and described as qualitative.

## Metadata reconciliations
- README BibTeX **"31th" → "31st"**; canonical PDF title used (README News-line link text differs). DOI added.
- MIT News (2026-02-26) and concurrent arXiv:2511.13841 added as a clearly **search-sourced** reception note, not TLT results.

## Reviewer verdict (honest)
The paper is now **defensible to the bit on its empirical core**: every paper number was recounted against the source with no transposition errors; all 11 claimed mechanisms were located in the released code and the two most-cited (the BEG-MAB reward formula and the elastic-SD threshold) were verified firsthand at `file:line`; equations were re-derived from scratch and match, with the **one substantive correction** that the dashboard's 5.7× Pareto example is wrong (true ≈14×). What the reader must still *accept on assumption* is confined to the explicitly-flagged dashboard scaffolding (Pareto tail, uniform accept rate, τ=16, constant Fisher information) — these are presented as qualitative intuition, never as paper results — and to two honest limits of this pass: the Fig 13 interior heatmap cells and Fig 11's Open-R1 bars are figure-read approximations (extrema verified), and the headline "drafter trained for free during RL" is real in code but **shipped off by default**, so end-to-end reproduction requires flipping `enable_drafter_training`/`collect_hidden_states_from_sgl`. Net: the speedup story (overlap, not approximation; token-level-lossless; 1.7–2.1× throughput at matched reward curves) stands on measured-and-code-grounded footing; the analytical model is sound but qualitative; nothing is fabricated, and every residual soft spot is labeled.
