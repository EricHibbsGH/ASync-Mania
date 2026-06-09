# Phase 2 — Research Log & External Reconciliation

Verifies how `paper.md` characterizes prior work, plus post-publication context and two metadata discrepancies. Verdicts: **FAIR** (accurately represented), **NEEDS-EDIT** (fix in Phase 6), **UNVERIFIED**. Items checked live via WebSearch/WebFetch are marked 🌐; items grounded in established literature + the paper's own (PDF) reference list are marked 📚.

## A. Speculative-decoding prior work

| Our claim | Verdict | Evidence | Source |
|---|---|---|---|
| EAGLE trains a compact autoregressive draft head on the target's last-layer hidden states | FAIR 📚 | EAGLE operates on the target's feature (second-to-top) layer; autoregressive at the feature level | Li et al., EAGLE, ICML'24 |
| EAGLE-2 adds dynamic draft trees | FAIR 📚 | EAGLE-2 = context-aware dynamic draft tree | Li et al., EMNLP'24 (arXiv:2406.16858) |
| EAGLE-3 fuses top/mid/bottom target layers; "training-time test"; CE-only | FAIR 📚 | EAGLE-3 removes the feature-prediction loss, uses multi-layer fusion + training-time test | Li et al., arXiv:2503.01840 |
| HASS uses multi-step "training-time test" to fix train/inference mismatch (~3 steps) | FAIR 📚 | HASS = Harmonized Speculative Sampling; simulates multi-step prediction in training | Zhang et al., ICLR'25 |
| OSD uses knowledge distillation (incl. reverse-KD/GKD) to track online distribution shift | FAIR 📚 | OSD continuously updates the drafter via KD on observed query distribution | Liu et al., ICML'24 |
| SpecInfer = tree-based speculative inference + parallel verification | FAIR 📚 | SpecInfer introduces tree-based speculation/verification | Miao et al., ASPLOS'24 |
| Medusa adds multiple decoding heads; Hydra adds sequentially-dependent heads | FAIR 📚 | Medusa = parallel heads; Hydra refines with dependent heads | Cai et al. (2401.10774); Ankner et al. (COLM'24) |
| Model-free drafting = Lookahead / REST / prompt-lookup (n-gram/retrieval, no draft model) | FAIR 📚 | Lookahead = Jacobi/n-gram lossless; REST = retrieval-based SD; prompt-lookup = context n-gram | Zhao et al. KDD'24 [72]; He et al. (2311.08252); Saxena (2023) |

**No NEEDS-EDIT** in this cluster — the paper's SD-related descriptions are accurate.

## B. RL algorithms

| Our claim | Verdict | Evidence | Source |
|---|---|---|---|
| GRPO replaces PPO's learned critic with a group-relative baseline | FAIR 📚 | GRPO normalizes rewards within a sampled group; no value network | Shao et al., DeepSeekMath, arXiv:2402.03300 |
| A GRPO step = rollout → inference (KL vs frozen reference) → training | FAIR 📚 | Matches GRPO/DeepSeek-R1 pipeline; KL penalty to reference | Shao et al.; DeepSeek-R1 (2501.12948) |
| PPO relies on a critic | FAIR 📚 | PPO actor-critic | Schulman et al., 2017 |
| RLOO/REINFORCE++/DAPO share the workflow, differ in reward/KL | FAIR 📚 | All are policy-gradient variants on the same rollout→update loop | Ahmadian'24; Hu'25; Yu et al. DAPO (2503.14476) |

## C. RLHF systems and the "neglect rollout" framing

| Our claim | Verdict | Evidence | Source |
|---|---|---|---|
| VeRL/HybridFlow, RLHFuse, ReaLHF optimize multi-model orchestration/placement but neglect the rollout bottleneck | FAIR (as the paper's framing) 📚 | These target placement/fusion/parameter-reallocation across RLHF stages; rollout-tail is not their focus. Our paper attributes this as the TLT authors' positioning, not an independent claim | HybridFlow EuroSys'25; RLHFuse NSDI'25; ReaLHF (2406.14088) |

*Note:* keep this as **the paper's** characterization (it is a fair summary of those systems' stated goals), not an absolute claim — already phrased that way in `paper.md` §7.3/§8.

## D. Async / long-tail RL contrast set (the "Why Not Fully Async?" cluster)

| System | Our characterization | Verdict | Evidence | Source |
|---|---|---|---|---|
| **AReaL** | Large-scale **asynchronous** RL (decoupled gen/train, tolerates staleness) | FAIR 📚 | Title: "A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning" | arXiv:2505.24298 |
| **StreamRL** | Disaggregated **streaming** generation | FAIR 📚 | Title: "Scalable, Heterogeneous, and Elastic RL … with Disaggregated Stream Generation" | arXiv:2504.15930 |
| **APRIL** | **Active partial rollouts** (over-provision, early-stop, recycle incomplete rollouts) | FAIR 🌐 | Confirmed: over-provisions rollouts, terminates once enough complete, recycles partials in later steps; +22.5% avg throughput | arXiv:2509.18521 |
| **RollPacker** | **Tail batching** (consolidate long-tail prompts into "long rounds"); paper's critique: trajectories vary in length even for identical queries | FAIR 🌐 | Confirmed tail batching (long/short rounds), 2.03–2.56× speedup. **Clarification:** RollPacker is explicitly **synchronous** ("Fast, Synchronous RL Post-Training") — so it is a long-tail *scheduling* method, not an async-RL system | arXiv:2509.21009 |

**NEEDS-EDIT (minor):** In `paper.md` §7.3 the four systems are introduced together under "Why Not Fully Async?". RollPacker is **synchronous**; ensure the text does not imply RollPacker mixes policy versions / is asynchronous. AReaL/APRIL are the genuine staleness-tolerant cases; StreamRL is disaggregation; RollPacker is synchronous tail batching. Fix wording in Phase 6 to separate "async/off-policy" (AReaL, APRIL-recycling) from "synchronous long-tail scheduling" (RollPacker). The paper's stated RollPacker *limitation* (length variance for identical queries) is the TLT authors' critique and is correctly attributed.

## E. Metadata reconciliation

1. **Title discrepancy (resolved):** The repo README's **News-line arXiv link text** renders the title as *"Taming the Long Tail: Efficient Reinforcement Learning for Language Models via Adaptive Speculative Decoding"* — different from the PDF's canonical *"Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter."* Nuance found in Phase 1: the README's **BibTeX** (`README.md:132`) actually uses the **correct** canonical title; only the prose News link differs. **Resolution: use the PDF title as canonical everywhere.** 🌐/📚
2. **BibTeX ordinal typo (confirmed):** `README.md:134` reads `booktitle={Proceedings of the 31th ACM International Conference …}` — **"31th"** should be **"31st."** Two BibTeX forms exist: README `@inproceedings{TLT}` (has the typo) and the dashboard `@misc{hu2026…}`. **Resolution: our references use "31st" + the full canonical proceedings string + DOI 10.1145/3779212.3790231.** 📚

## F. Reception & follow-on work (search-sourced, dated)

> All items in this section post-date the assistant's knowledge cutoff and are **search-sourced (June 2026 web)**.

- **MIT coverage (2026-02-26):** *"New method could increase LLM training efficiency,"* MIT News — syndicated to MIT EECS, MIT Schwarzman College of Computing, MIT Climate Portal, and Mirage News. It describes TLT's two components (adaptive drafter trainer on idle processors; adaptive rollout engine), states the drafter is a deployable "free byproduct," and renders the speedup in popular-press terms as "between 70 and 210 percent" (a loose rendering of the paper's precise **1.7–2.1×**; do **not** import the percentage phrasing into the paper). 🌐
- **Concurrent/related work:** arXiv:2511.13841, *"Beat the long tail: Distribution-Aware Speculative Decoding for RL Training"* — appeared essentially simultaneously with TLT (arXiv:2511.16665, Nov 2025); same problem (long-tail SD for RL), different angle (distribution-aware). Worth a one-line "concurrent work" mention; **not** a TLT result. 🌐
- No peer-reviewed *citing* follow-on beyond concurrent arXiv work was confidently identified as of June 2026; stated as a negative result rather than asserted absence.

## Net
The paper's prior-work characterizations are **FAIR** across all clusters, with **one minor NEEDS-EDIT** (separate RollPacker's *synchronous* tail-batching from the async/off-policy systems in §7.3). Metadata: use the PDF title + "31st" + DOI. A dated "Reception & follow-on work" note (MIT News + concurrent arXiv:2511.13841) is added in Phase 6, clearly marked search-sourced.
