# CHANGELOG — Review Pass (Phases 0–7)

Every change from the prior `paper.md` draft, grouped by type, each with its source. Provenance tags: [Paper], [Dashboard], [Code: path], [Author synthesis].

## Code-grounded (new [Code] axis added throughout §4–§5, §8)
- **Spot Trainer → `EagleBackgroundTrainer`** named and cited. [Code: verl/workers/drafter/eagle_background_trainer.py:35]
- **Worker states: 5, not 3.** Corrected "BUSY/IDLE/TRAINING" to note the code's `IDLE/GENERATING/RELEASED/TRAINING/COMPLETED` (BUSY=GENERATING, IDLE=RELEASED). [Code: worker_manager.py:54-61]
- **Coordinator transport** = ZeroMQ REP/PUB + REQ/SUB, rank-0. [Code: worker_manager.py:91-107,441]
- **Promotion default `min_workers_for_training=1`.** [Code: fastrl_trainer.yaml:25]
- **Drafter cadence `training_interval_steps=10`** (shipped YAML) — reconciles paper's "~10 steps." [Code: fastrl_trainer.yaml:20]
- **DataBuffer one-step-offset** = collect at `(step+1)%interval==0`, train from last `buffer_steps=2`. [Code: worker_manager.py:490-493]
- **DataBuffer caveat:** `random.sample` over a 512-token window, **no explicit long-seq prioritization** (softer than paper). [Code: eagle_background_trainer.py:299-356]
- **Selective ckpt** = `dcp.async_save` + frozen set `{embed_tokens.weight, lm_head.weight}`. [Code: eagle_background_trainer.py:83,119-148]
- **Drafter loss** = `SmoothL1Loss` + soft-CE, 0.5/0.5 (refines "L1+CE"). [Code: eagle_background_trainer.py:73,558-578]
- **Single-layer drafter** `num_hidden_layers=1`; embed frozen both paths; **LM head frozen on EAGLE, trainable on EAGLE-3.** [Code: eagle_trainer.py:190,255-258; eagle3_trainer.py:475]
- **Elastic threshold = 32 + 10-check warmup hysteresis.** [Code: grpo_7B.sh:28; eagle_worker.py:361,196]
- **BEG-MAB exact reward** `a=Σacc/bs+1`, `r=a·bs/elapsed`; **ε=0.1, window=1000, buckets [1,2,5,21], OOM bound draft·bs≤2048.** [Code: eagle_worker.py:1234,1240; eagle_mab.py:210,220,442]
- **CUDAGraph bucket schedule** (exact granularities); draft=DECODE, target=TARGET_VERIFY. [Code: server_args.py:789-796]
- **N-gram drafter defaults** window=12, branch=18. [Code: server_args.py:342-347]
- **GRPO run config** n=8, KL `low_var_kl`/0.001 (not in reward), LR 1e-6, TP 4/8, SP 4/8 added to §2.1/§6.1/§8. [Code: examples/grpo_*.sh]
- **Online-training off by default** (`enable_drafter_training=false`) — surfaced as the top reproducibility caveat. [Code: fastrl_trainer.yaml:19,26]
- New `data_table.md` section (C1–C20) carrying the [Code] column.

## Numeric corrections
- **Dashboard "5.7× vs median" is wrong → corrected to ≈14×** (exact 20.07·x_min / MC 20.14 / median √2). Origin of 5.7 identified as √128/2 = B^(1/α)/α (mis-normalization). [Author synthesis; recomputed]
- **Dashboard "1.08× at BS=32"** reaffirmed inconsistent with Tbl 4 (true 1.70×); kept flagged, not propagated. [Paper Tbl 4]
- **9.2× checkpoint** re-framed as an end-to-end measurement (not a 7B param-ratio; the drafter's frozen embed/LM-head dominate). Corrected a sub-agent's mis-estimate. [Paper Fig 17a; Code]
- Geometric series, speedup interior-optimum, KL recursion **re-derived and verified** (α→1 ⇒ γ+1 = 9.000; caps; argmax-γ = 3/5/7). [Author synthesis]

## Quote fixes (carried + re-audited)
- Removed non-contiguous quote "∼85% of the total step time" → paraphrase. (prior pass)
- De-duplicated "in a non-blocking manner" to one use; "free byproduct" re-cited to Abstract. (prior pass)
- All 15 remaining prose quotes re-audited: verbatim, ≤ 8 words, ≤ 15-word rule satisfied. [phase3_accuracy.md §4]
- "Lossless" restated precisely as **token-level distributional equivalence** in the provenance block, Abstract, §2.2, §7.2. [Author synthesis / R2]

## New figures (Phase 4; `make_figures.py` → `figures/*.pdf|svg`)
- 11 vector figures added + embedded in a new **Figures** section with provenance captions; `figures/manifest.md` records exact-vs-schematic.
- Exact [Paper]: `fig11` (TLT/VeRL/TLT-Base), `tbl4`, `tbl2`, `tbl3`, `fig17`, `tbl6`, boxed extrema of `fig13`.
- Approx/figure-read: `fig13` interior cells, `fig11` Open-R1 bars. Schematic: `fig16_*`, `fig1_*`. Modeled [Dashboard]: `dashboard_markov_surface`, `dashboard_threshold_curve`.

## Limitations added (red-team R1–R12 → §9 Threats to Validity)
- Throughput ≠ convergence/accuracy (R3); GPU-diversity is BS1/TP1 micro-benchmark (R4); online-training off by default (R5); uniform-α model is qualitative (R6); 9.2× is end-to-end (R7); drafter is "adequate," not SOTA standalone (R8); BEG-MAB exploration/non-stationarity (R9); workload-dependent gains (R10); RollPacker is synchronous (R11); temp-0 SD grids vs temp-0.9 training (R12). New §9 consolidates the assumptions ledger.

## Citation fixes (Phase 5)
- **BibTeX "31th" → "31st"**; switched to `@inproceedings` with full proceedings string + **DOI 10.1145/3779212.3790231**; canonical PDF title. [README.md:134 typo; Paper]
- **Title discrepancy** noted: README News-line link uses a non-canonical title; canonical PDF title used. [README; Paper]
- Dashboard cited with **access date 2026-06-05**; repo cited as an artifact with commit `bce3df7`, Apache-2.0.
- Added **MIT News (2026-02-26)** and concurrent **arXiv:2511.13841** under a clearly search-sourced reception note (not TLT results).
- §7.3 reorganized to separate off-policy (AReaL, APRIL) / disaggregation (StreamRL) / synchronous scheduling (RollPacker).

## Structural
- Renumbered: new **§8 Implementation & Reproducibility**, **§9 Threats to Validity / Assumptions Ledger**; Extensions → §10, Conclusion → §11.
- Provenance block upgraded to **four axes** (added [Code]); "lossless" and "throughput" definitions pinned.
