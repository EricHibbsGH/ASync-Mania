# Review Report — IEEE refactor of "Overlapping, Not Approximating"

Deliverables in this folder:

| File | What it is |
|---|---|
| `main.pdf` | Compiled IEEE conference paper (IEEEtran, two-column, 10 pp. incl. references). Zero LaTeX errors, zero overfull boxes, all fonts embedded (no Type 3). |
| `main.tex`, `refs.bib`, `main.bbl` | LaTeX source + bibliography (IEEEtran.bst). Upload all three plus `figures/` to arXiv. |
| `make_figures_ieee.py` | Regenerates `figures/*.pdf` from inline, audited data arrays (corrected values). |

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Figures: `python make_figures_ieee.py`.

Verification sources used: the original TLT PDF (arXiv:2511.16665v2, downloaded 2026-11-19 and read page by page); the `mit-han-lab/fastrl` repository checked out at commit `bce3df7` (the current HEAD); arXiv metadata for 12 cited preprints; and Python recomputation of every ratio, mean and model value, including Monte Carlo.

---

## 1. Numerical and technical verification checklist

Legend: ✅ verified · ❌ error found and corrected · ⚠️ source-level inconsistency or approximation, flagged in the paper.

### 1a. End-to-end results (TLT Fig. 11)

| Item | Original manuscript | Source (arXiv v2) | Recomputed | Status |
|---|---|---|---|---|
| H100 TLT: Qwen-7B / DS-7B / Qwen-32B / Llama-70B / geomean | 1.76 / 1.71 / 2.12 / 1.86 / 1.86 | same | geomean 1.856 | ✅ |
| H100 TLT-Base | 1.42 / 1.38 / 1.54 / 1.31 / 1.41 | same | geomean 1.410 | ✅ |
| **A100 TLT** | 1.79 / 1.66 / 2.07 / 1.73 / **1.70** | **1.73 / 1.70 / 2.07 / 1.66 / 1.79** | geomean 1.783 | ❌ fixed: per-model labels were shifted, and the true geomean is **1.79×**, not 1.70× |
| **A100 TLT-Base** | 1.35 / 1.23 / 1.49 / 1.31 / 1.38 | **1.38 / 1.31 / 1.49 / 1.23 / 1.35** | geomean 1.349 | ❌ fixed (same shift) |
| Open-R1 bars | H100 Llama 0.18, geomean 0.16; A100 Qwen-32B n/a | Llama-70B: **no bar** on either GPU; geomeans 0.18 (H100) / 0.17 (A100); A100 Qwen-32B **0.06** | — | ❌ fixed |
| "1.7–2.1×" range | stated | caption says 1.7–2.1× | actual min is 1.66× (Llama-70B, A100) | ⚠️ rounding, now disclosed in Threats |
| Abstract "over 1.7×" | ✅ | ✅ | — | ✅ |

Because the A100 numbers sat in the old abstract ("1.70× on A100"), the most important correction in this review is that **the A100 geomean is 1.79×**.

### 1b. Other measured values

| Item | Check | Status |
|---|---|---|
| Rollout ≈85% of step | TLT §1(1), Fig. 1a | ✅ |
| ByteDance trace: Qwen2.5-32B, 128×H20-96GB, 11 days, 385 steps, ≈40 min/step, eval every 5 steps ≈20 min, max length 20,480 | TLT §1(3), Fig. 2 | ✅ (note: 385×40 min = 10.7 d, and adding 77 evals×20 min gives 11.8 d, so the source's "11 days" is approximate ⚠️) |
| Table 1 topK sweep: accept 8.29–8.67, speedup 3.47–3.65× | min 3.47 (topK 16), max 3.65 (topK 6) | ✅ |
| Fig. 13 extrema 3.66× (d12/v64), 9.78 (d16/v128) | circled in the original figure | ✅ |
| **Table 1 topK=8 (3.64×) vs Fig. 13b d12/v64/topK8 (3.66×)** | same nominal config, two values | ⚠️ source inconsistency, disclosed |
| Table 4 (all 24 cells) | the BS=1 row equals Fig. 13b's depth-10 row (3.22/3.46/3.56/3.62), which confirms internal consistency | ✅ |
| Table 2 ratios | 605.05/259.71=2.330; 430.24/164.65=2.613; 259.01/92.83=2.790; 293.84/100.89=2.912; 187.44/65.28=2.871; 166.41/51.75=3.216 | ✅ |
| "More I/O-bound GPUs gain more" | RTX 5090 2.91 > RTX 4090 2.87, so the trend is not monotone | ⚠️ wording softened to "tend to" |
| Table 3 scaling | 1.21/1.45/1.62/1.76; OOM/OOM/1.83/2.12 | ✅ |
| Table 5 CUDAGraph | 30.39/10.69 = 2.843 → "2.8×" ✅; bucketed/single = **1.37×** and naïve/single = 3.89× (new, derived). The original said "near single-strategy cost", which overstated it: bucketing still costs +37% | ❌ wording fixed |
| Fig. 17a checkpoint | 893/280 = 3.19 (3.2×); 893/97 = 9.21 (9.2×); 280/97 = 2.89 (selectivity share, derived) | ✅ |
| Fig. 17b packing | 29.6/13.3 = 2.23 (2.2×) | ✅ |
| Fig. 14 case study | 337 s / 138 s = 2.44× (read from the original figure) | ✅ (newly cross-checked) |
| Table 6 | (6.53−4.59)/4.59 = **+42.3%**; downstream (5.15−3.76)/3.76 = **+37.0%** (new, derived) | ✅ |
| Table 7 | 542.12/242.11 = 2.239; 553.91/242.11 = 2.288; 617.42/242.11 = 2.550 | ✅ |
| Table 8 | all six accept/throughput pairs re-transcribed | ✅ |
| Overheads (<1%, <3 s, every 10 steps) | TLT §6.4 | ✅ |
| Testbed (8×DGX H100, 2×Xeon 8480C, 2 TB, 900 GB/s NVLink, 400 Gb/s IB) | TLT §6.1 | ✅ (units checked: GB/s vs Gb/s correct) |
| Metric definition | token throughput per RL step, averaged over 3 steps after warm-up, **with a trained draft model** | ✅ (newly added; it matters for the reproducibility caveat) |
| Fig. 16 "≈32% vs ≈6% at index 8" (dashboard reading) | Original figure at index 8 reads ≈25% vs ≈3%; ≈32%/≈6% matches **index 7** | ⚠️ removed from the paper (figure-read, likely off by one index) |

### 1c. Equations and models

| Item | Recomputed | Status |
|---|---|---|
| Exact Pareto E[M_B], α=2, B=128 | 20.073·x_min; Monte Carlo 20.142; asymptotic 20.053 | ✅ |
| Max/median ratio | 20.07/1.414 = **14.19×** | ✅ |
| Dashboard closed form | 22.63·x_min → 16.0× median | ✅ |
| Dashboard "5.7×" | = √128/2 = 5.657, a normalization error | ✅ correction retained |
| E[n] = (1−α^{γ+1})/(1−α) limits | α→1: 9.000 at γ=8; γ→∞: 5.0 at α=0.8 | ✅ |
| argmax_γ speedup = 3/5/7 for c = 4/8/16 | holds at **α = 0.8**; the original omitted the α condition, which is now stated | ✅ |
| k·b(k) peak at k=τ | d/dk = e^{−k/τ}(1−k/τ) = 0; peak value τ·b₀/e = 5.89 matches the figure | ✅ |
| Drift inequality ηI > δ | "iff" softened to "to first order … when", because the O(η²) term makes "iff" too strong | ❌ fixed |
| Amdahl (new) | ceiling 1/(1−0.85) = 6.67×; implied s: 1.66→1.88, 2.12→2.64, 1.86→2.19, 1.79→2.08 | ✅ |
| Step-time reduction 1−1/S (new) | 1.7→41.2%, 1.86→46.2%, 2.12→52.8% | ✅ |
| GPU-hours (new) | 128×11×24 = 33,792; savings 33,792×(1−1/1.7) = 13,914 to ×(1−1/1.86) = 15,624 → "≈14,000–16,000" | ✅ (illustrative, labeled) |

### 1d. Code claims (re-verified at `fastrl@bce3df7`)

| Claim | Location | Status |
|---|---|---|
| `enable_drafter_training: false`, `collect_hidden_states_from_sgl: false` | `verl/trainer/config/fastrl_trainer.yaml:19,26` | ✅ |
| `bs_threshold: 32`, `training_interval_steps: 10`, `min_workers_for_training: 1`, MAB thresholds [1,2,5,21] | same YAML :9, :20, :25, :17 | ✅ |
| 5-state `WorkerState` enum | `worker_manager.py:54–61` | ✅ |
| `should_collect_data_this_step` returns False unless the drafter is trained; `(step+1) % interval == 0` | `worker_manager.py:489–493` | ✅ (strengthens the caveat) |
| `STOP_TRAINING` / `BATCH_COMPLETE` | `worker_manager.py:78–79, 221–225` | ✅ |
| `EagleBackgroundTrainer` :35; `SmoothL1Loss` :73; frozen names :83; `dcp.async_save` :119–148; `buffer_steps=2` :276; `random.sample` :302; checkpoint cadence `step//100` :596 | `verl/workers/drafter/eagle_background_trainer.py` | ✅ |
| `num_hidden_layers = 1` :190; `requires_grad = False` :256, :258 | `eagle-train/eagle_trainer.py` | ✅ |
| EAGLE-3 LM head trainable (comment at ~:474–475) | `eagle-train/eagle3_trainer.py` | ✅ |
| Warm-up checks = 10 (:196), enable check (:364) | `sglang/srt/speculative/eagle_worker.py` | ✅ |
| Reward `a = sum(accept_len)/bs + 1`, `reward = stable_accept_length*bs/total_time` | `eagle_worker.py:1234–1240` | ✅ |
| `BucketedEpsilonGreedyMAB` :197; ε = 0.1 :210; window 1000 :111/:381; OOM guard ≤ 2048 :442 | `sglang/srt/speculative/eagle_mab.py` | ✅ |
| n-gram window 12, branch 18 | `sglang/srt/server_args.py:342,346` | ✅ |
| CUDA-graph spec buckets | `server_args.py:789–796` | ✅ |
| GRPO n=8, KL `low_var_kl` 0.001, temp 0.9, TP 4, SP 4, 8 GPUs, 32K response | `examples/grpo_7B.sh` | ✅ |
| 32B: `#SBATCH --nodes=8`, TP 8, SP 8, max prompt 2048 | `examples/grpo_32B_multi_nodes.sh` | ✅ |
| Packing "attention-masked, no padding" | `eagle_background_trainer.py:410–417` concatenates with an **all-ones attention mask** and no position IDs (model call :504). `qwen2_eagle.py:88–89` defaults positions to one continuous range, `:92–114` builds a plain causal mask, and `:193–197` forces eager attention (FlashAttention-2 off) | ✅ now in the paper (Sec. IX, last paragraph): samples in a pack can attend to earlier samples (static reading, not measured) |
| Drafter training skipped unless hidden states are collected | `eagle_background_trainer.py:473–479` | ✅ added to Sec. IX; strengthens the "both flags" caveat |
| Tree drafting `eagle_worker.py:748–821` | `draft_forward` starts at 748 and returns at 821 | ✅ |

### 1e. Citations

| Item | Status |
|---|---|
| TLT title, 10 authors, ASPLOS '26 Vol. 2, Pittsburgh, Mar 22–26 2026, DOI 10.1145/3779212.3790231, arXiv v2 dated 21 Jan 2026 | ✅ (from the PDF's first page) |
| arXiv IDs, titles, first authors: AReaL, APRIL, RollPacker, StreamRL, PRIME/Eurus-2, EAGLE-3, DeepSeekMath, DeepSeek-R1, DAPO, REINFORCE++, Krell packing, "Beat the long tail" | ✅ via arXiv metadata. Fixes: **Krell is 2021, not 2022**; full titles now used for StreamRL, EAGLE-3 and REINFORCE++; "Beat the long tail" now credits **Z. Shao et al.** |
| Venue entries (ICML, ICLR, NeurIPS, OSDI, KDD, ACL, EuroSys) | standard and well known; author lists not machine-checked | ⚠️ spot-check |
| Self-citations (dashboard, repo) | URLs carried from your README | ⚠️ confirm live and public |
| References listed but never cited in text (Medusa, Hydra, SpecInfer, RLHFuse, ReaLHF, ZeroMQ, FSDP2, Qwen2.5 report, MT-Bench, Adam, W&B trace, MIT News) | IEEE lists only cited works | removed |

---

## 2. Claims, citations, and details requiring author confirmation

1. **Accept the A100 correction** (geomean 1.79×, per-model 1.73/1.70/2.07/1.66; TLT-Base 1.38/1.31/1.49/1.23/1.35). Then also fix the old sources so your public record is consistent: `README.md`, `analysis/data_table.md` rows 6–8, `build/make_figures.py` (the `A` dict and the Open-R1 arrays), `paper/paper.md`, and the GitHub Pages site and dashboard if they show these values.
2. ✅ **Author block:** "Independent Researcher, Atlanta, GA, USA" (added 2026-11-20). The printed email `ericbhibbs@gmail.com` is still yours to confirm or remove.
3. ✅ **Repository link (ref. [13]) now resolves** (HTTP 200, public, checked 2026-11-19). Two follow-ups remain. (a) The repo was last pushed 2026-06-09, so it still has the old A100 numbers and lacks this `ieee/` folder; push both before posting the preprint. (b) The GitHub Pages site `https://erichibbsgh.github.io/ASync-Mania/` still returns 404. The paper doesn't cite it, but the repo README does, so re-enable Pages under Settings → Pages (source: `main` / `docs`) or drop the link.
4. ✅ **Dashboard URL** (ref. [12]) resolves (HTTP 200), trailing `./` included.
5. ✅ **Title simplified at your request (2026-11-20):** "Speeding Up LLM Reinforcement Learning with Idle GPU Time: An Independent Code Audit of TLT".
6. **New derived analyses** (all arithmetic on cited inputs, labeled "derived"): the Amdahl consistency check (Sec. VI-E, Fig. 4), step-time reductions, the GPU-hour illustration (Sec. X), the +37.0% downstream gain, the 1.37×/3.89× memory ratios, and the 2.9× selectivity share. Confirm you're comfortable claiming them.
7. **Added citations:** PPO [9], Roofline [10], and Amdahl [16] support statements that were already in the text. Approve or remove them.
8. **Fig. 16 reading.** The dashboard's 32%/6% appears to be index 7, not 8. I removed the numbers. If you want them back, re-read the original figure at high zoom.
9. **New audit finding, now in the manuscript (Sec. IX, last paragraph, plus one sentence under Audit limits).** I traced the drafter's forward pass. Packed samples get an all-ones attention mask and continuous positions, then a plain causal mask, and FlashAttention-2 is switched off. So samples can attend across pack boundaries. It is written as an *observation*, not a sixth "divergence", because the TLT paper never claims boundary masking; the abstract's "five divergences" therefore still holds. It is a static reading only. Before citing it in interviews, run a tiny packed vs. unpacked forward pass of the drafter and confirm the logits differ. To remove it, delete that paragraph and the Audit-limits sentence.
10. **"Eleven mechanisms."** The count follows Table I's list; confirm it matches your Phase-1 definition.
11. **Table 1 vs Fig. 13 (3.64× vs 3.66×).** "Most likely separate runs" is an inference. Keep it hedged, or ask the TLT authors.
12. **CC-BY-4.0 reuse.** Fig. 1 is labeled "Adapted from [3, Fig. 1b]", and Fig. 3/Fig. 5 re-plot TLT data with attribution. That is compliant with CC-BY, but confirm you're comfortable with it.
13. **Removed content** (restore any you want): the MIT News coverage paragraph, the schematic accept-rate figure (invented interior points), the synthetic long-tail figure (random data that could be mistaken for measurements), the approximate heatmap interior cells, and the internal file references (`phase*.md`).
14. **Reward-curve window.** "Over the reported training window" refers to roughly 110 steps in TLT Fig. 12; confirm the wording is acceptable.

---

## 3. Summary of changes

**Format and structure**
- Converted to IEEEtran conference format: numbered sections, IEEE-style abstract and Index Terms, numbered equations, and table and figure captions in IEEE style. Figures and tables are numbered in order of first citation, with numeric citations carrying pinpoints (e.g., [3, Table 4]) and a BibTeX bibliography built with IEEEtran.bst, which orders references by first citation.
- Removed non-IEEE elements: table of contents, list of figures/tables, build stamp, the inline BibTeX listing, and the bracket provenance tags. The tags are replaced by a stated **four-class provenance method** (measured / code-confirmed / modeled / derived) in a new **Analysis Methodology** section, so the provenance discipline survives in reviewer-acceptable form.
- Removed a duplicated conclusion paragraph and leftover drafting artifacts ("post-cutoff", "search-sourced", "Phase 2 reconciliation", internal file names).

**Correctness**
- Corrected the A100 end-to-end and Open-R1 values, and with them the abstract claim (1.70× → 1.79×).
- Corrected "near single-strategy cost" (actually 1.37×), the missing α = 0.8 condition, "iff" → first-order "when", the Krell year, and incomplete reference titles and authors.
- Removed claims that couldn't be verified: the Fig. 16 percentages and "attention-masked" packing.
- Re-verified every code reference against `bce3df7`.

**Value and positioning (for AI PM, system-design, infra and ML-systems roles)**
- New **Fig. 1 system timeline** (TikZ), showing the overlap design at a glance.
- New **Table I** (mechanism → cost hidden → measured effect → exact code location), a system-design view a hiring manager can scan in 30 seconds.
- New **Amdahl consistency check** (Sec. VI-E, Fig. 4). It links rollout-level and end-to-end results and states the 6.7× ceiling, which is the kind of first-principles bound infra interviews look for.
- New **Practical Implications for Deployment** section (Sec. X): GPU-hour cost translation, adoption criteria, a code-confirmed configuration checklist, a memory budget, monitoring KPIs, and a risk profile.
- Rewrote the abstract, introduction and contribution list around auditability and adoptability, with the 33,792 GPU-hour production cost as motivation.
- Tightened the threats-to-validity section and added a source-consistency subsection, which signals rigor and honesty.

---

## 4. Final publication-readiness checks

1. Resolve items 1–4 in section 2 (A100 fix propagated, author block, repo public, URLs live), then rebuild with the four-step command above.
2. **arXiv:** upload `arxiv_submission.zip` (already built: `main.tex`, `main.bbl`, `refs.bib`, `figures/*.pdf`; test-compiled in an empty folder with 0 errors and 0 undefined citations). Rebuild the zip if you change anything. Suggested primary category **cs.DC**, cross-listed to **cs.LG**. First-time submitters may need an endorsement for the category.
3. Run a spell and grammar pass, and click every URL in the compiled PDF.
4. If you later submit to a peer-reviewed IEEE workshop or conference: check its page limit (the body is about 9 pages), anonymize if it's double-blind (remove your name and refer to the dashboard and repo in the third person), validate the PDF with IEEE PDF eXpress, and complete the IEEE copyright form. Fonts are already fully embedded.
5. Before posting, commit the `ieee/` folder to the ASync-Mania repo so "all analysis artifacts are publicly released" is true on the day you publish.
6. For job applications, link the arXiv abstract page (not the PDF) on your resume and LinkedIn, with a one-line result such as: "Code-audited an ASPLOS '26 RL-training system; found that its headline online-drafter feature ships disabled; built an Amdahl model showing a 6.7× ceiling."
