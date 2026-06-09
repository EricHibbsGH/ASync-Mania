# Consolidated Data Table — Audit Trail

Every quantitative claim used in `paper.md`, with its exact source location and a provenance tag.

**Tag legend**
- **[P]** = Paper (TLT, ASPLOS '26, arXiv:2511.16665v2). Source given as §/Fig/Tbl.
- **[D]** = Dashboard (Eric Hibbs, *Interactive companion*, "Model Math" §1–6). These are the **dashboard author's** derivations/assumptions, **not** paper results.
- **[S]** = Author synthesis (arithmetic or framing I add on top of P/D; underlying inputs are cited).

> Conditions (model, GPU, TP, batch size, draft depth, etc.) are carried inline with each value. Where the paper and another part of the paper differ, both are recorded.

| # | Category | Claim / Value (with units & conditions) | Exact Source Location | Tag |
|---|----------|------------------------------------------|------------------------|-----|
| 1 | End-to-end | "over 1.7×" end-to-end RL training speedup over SOTA | Abstract | P |
| 2 | End-to-end | "1.7-2.1×" speedup over SOTA system VeRL | Fig 11 caption | P |
| 3 | End-to-end | Rollout ≈ 85% of total RL step time | §1, item (1): "(∼85%) of the total step time"; Fig 1a | P |
| 4 | End-to-end (H100) | TLT vs VeRL normalized throughput: Qwen-7B 1.76×, DeepSeek-7B 1.71×, Qwen-32B 2.12×, Llama-70B 1.86×, Geomean 1.86× (VeRL=1.00) | Fig 11 (left, H100) | P |
| 5 | End-to-end (H100) | TLT-Base vs VeRL: Qwen-7B 1.42×, DeepSeek-7B 1.38×, Qwen-32B 1.54×, Llama-70B 1.31×, Geomean 1.41× | Fig 11 (left, H100) | P |
| 6 | End-to-end (A100) | TLT vs VeRL: Qwen-7B 1.79×, DeepSeek-7B 1.66×, Qwen-32B 2.07×, Llama-70B 1.73×, Geomean 1.70× | Fig 11 (right, A100) | P |
| 7 | End-to-end (A100) | TLT-Base vs VeRL: Qwen-7B 1.35×, DeepSeek-7B 1.23×, Qwen-32B 1.49×, Llama-70B 1.31×, Geomean 1.38× | Fig 11 (right, A100) | P |
| 8 | End-to-end | Open-R1 normalized throughput well below VeRL (≈0.06–0.25× of VeRL across models; exact per-bar values partly illegible in figure) | Fig 11 (both panels) | P (approx from figure) |
| 9 | Peak | Peak end-to-end speedup 2.12× on Qwen-32B / H100 (and 8 nodes) | Fig 11; Tbl 3 | P |
| 10 | ByteDance trace | Model Qwen2.5-32B; hardware H20 96GB GPUs | Fig 2 caption | P |
| 11 | ByteDance trace | 128 GPUs, 11 days, 385 steps | §1 item (3); Fig 2 | P |
| 12 | ByteDance trace | ≈40 min average per training step | §1 item (3): "averaged approximately 40 minutes" | P |
| 13 | ByteDance trace | Evaluation every 5 steps, ≈20 min each | §1 item (3) | P |
| 14 | ByteDance trace | Customized max length 20,480 tokens | Fig 2 annotation | P |
| 15 | ByteDance trace | Reported percentiles: Max, Q3 (p75), Median (p50); finer percentiles not in source | Fig 2 caption | P |
| 16 | Self-collected trace | Fig 1a max length 30K; models Qwen-7B & DeepSeek-R1-7B | Fig 1a | P |
| 17 | Rollout length | Reasoning rollouts "over an order of magnitude longer" than typical RLHF outputs | §1 | P |
| 18 | Drafter arch | Single trainable transformer decoder layer; reuses (frozen) target Embedding + LM Head | §4.1; Fig 6/7 | P |
| 19 | Drafter arch | Trainable params ≈ 1/layer_num of target total params | §4.1: "approximately 1/layer_num" | P |
| 20 | Drafter arch | Qwen2.5-32B has 64 layers; Qwen2.5-0.5B has 24 layers; param count reduced 64× | §4.1 | P |
| 21 | Drafter speed | TLT single-layer drafter (≈ same param count as Qwen2.5-0.5B drafter) is 2.4× faster | §4.1: "significantly faster (2.4×)" | P |
| 22 | Drafter size | Target model "over 20×" larger than draft model | §4.1 (Training Process) | P |
| 23 | Drafter strategies | Eagle: last-layer hidden states, L1+CE loss, no train-time test; HASS: last-layer, L1+CE, 3-step train-time test; Eagle-3: top/mid/bottom layers, CE-only, 7-step | Fig 7 (lower table) | P |
| 24 | Elastic SD | SD activated only when remaining request count < threshold; default 32 | §5.1 "Adaptivity is Necessary"; Fig 14 | P |
| 25 | Tree decoding | Example config topK=2, Draft_Depth=4, Tokens_to_Verify=8 | Fig 9 caption | P |
| 26 | CUDAGraph (illustrative) | Single strategy Mem 1× / Thpt 1×; Multiple Mem 4× / Thpt 1.8×; Bucketed Mem 1.2× / Thpt 2× (values "illustrative") | Fig 10 | P (illustrative) |
| 27 | BEG-MAB | Arm = (Draft_Depth, topK, Tokens_to_Verify); reward = accepted_token_num / step_latency | §5.2; Alg 1 lines 8–9 | P |
| 28 | topK sweep | Qwen-32B, H100, TP=4, Draft_Depth=12, Tokens_to_Verify=64. topK 4/6/8/10/12/16 → Accept Len 8.29/8.66/8.67/8.67/8.60/8.42 | Tbl 1 | P |
| 29 | topK sweep | Same config → Speedup 3.51×/3.65×/3.64×/3.64×/3.56×/3.47× | Tbl 1 | P |
| 30 | SD grid | Qwen-32B, TP=4, BS=1, topK=8, temp=0, CUDAGraph on: peak speedup 3.66× at Draft_Depth=12, Tokens_to_Verify=64 | Fig 13(b) | P |
| 31 | SD grid | Same grid: max average accept length 9.78 at Draft_Depth=16, Tokens_to_Verify=128 | Fig 13(a) | P |
| 32 | Batch sensitivity | Qwen-32B, H100, TP=4, Draft_Depth=10, topK=8. Speedup by Tokens_to_Verify {16,32,48,64}: BS=1 → 3.22/3.46/3.56/3.62 | Tbl 4 | P |
| 33 | Batch sensitivity | BS=2 → 3.08/3.28/3.39/3.38; BS=4 → 3.01/3.09/3.13/2.98 | Tbl 4 | P |
| 34 | Batch sensitivity | BS=8 → 2.73/2.63/2.51/2.27; BS=16 → 2.67/2.52/2.24/1.91 | Tbl 4 | P |
| 35 | Batch sensitivity | BS=32 → 2.48/2.23/1.90/1.70 (best at fewest verify tokens) | Tbl 4 | P |
| 36 | Batch sensitivity | Insight: larger batches achieve optimal performance when *fewer* tokens are verified; SD still benefits even at BS=32 | §6.3 text; Tbl 4 | P |
| 37 | GPU diversity | Qwen2.5-7B, BS=1, TP=1, rollout tok/s (w/SD, w/o SD, speedup): B200 605.05 / 259.71 / 2.33× | Tbl 2 | P |
| 38 | GPU diversity | H100 430.24 / 164.65 / 2.61×; A100 259.01 / 92.83 / 2.79× | Tbl 2 | P |
| 39 | GPU diversity | RTX 5090 293.84 / 100.89 / 2.91×; RTX 4090 187.44 / 65.28 / 2.87×; RTX 3090 166.41 / 51.75 / 3.22× | Tbl 2 | P |
| 40 | GPU diversity | Trend: more I/O-bound (consumer) GPUs gain more from SD | §6.3; Tbl 2 | P |
| 41 | Cluster scale | End-to-end speedup, nodes {1,2,4,8}: Qwen2.5-7B → 1.21×/1.45×/1.62×/1.76× | Tbl 3 | P |
| 42 | Cluster scale | Qwen2.5-32B → OOM / OOM / 1.83× / 2.12× | Tbl 3 | P |
| 43 | Bucketed CUDAGraph | Llama-3-8B, H100, TP=4, 4 strategies. Single 7.81 GB; Vanilla Multiple 30.39 GB; Bucketed 10.69 GB | Tbl 5 | P |
| 44 | Bucketed CUDAGraph | 2.8× memory reduction vs naive multi-strategy (30.39/10.69 = 2.84) | §6.3 text; Tbl 5 | P |
| 45 | Adaptive drafter | Accept length, RL Training: Target-Base 4.59 → Target-R 6.53 | Tbl 6 | P |
| 46 | Adaptive drafter | Accept length, Downstream (MTBench math/coding/reasoning): Target-Base 3.76 → Target-R 5.15 | Tbl 6 | P |
| 47 | Adaptive drafter | Target-Base = Qwen2.5-7B; Target-R = model after RL | Tbl 6 caption | P |
| 48 | Accept-rate decay | Fig 16: vanilla drafter accept rate collapses at distant token indices; adaptive sustains higher rate (indices 1–8) | Fig 16 | P |
| 49 | Accept-rate decay | Dashboard reads Fig 16 as ≈32% (adaptive) vs ≈6% (vanilla) accept rate at draft-token index 8 | Dashboard "Model Math" (Fig 16 reading) | D |
| 50 | Adaptive drafter | +42% RL-training accept-length gain from co-training = (6.53−4.59)/4.59 = 42.3% | Tbl 6 (inputs); arithmetic | S / D |
| 51 | SD methods | Qwen2.5-7B, TP=2, BS=1, H100; training cost normalized to per-step Eagle. Base(No-SD): accept 1.00, 242.11 tok/s, 1.00× | Tbl 7 | P |
| 52 | SD methods | HASS: 6.67, 553.91 tok/s, 2.29×, 3× train cost | Tbl 7 | P |
| 53 | SD methods | Eagle-3: 6.83, 617.42 tok/s, 2.55×, 7× train cost | Tbl 7 | P |
| 54 | SD methods | Eagle: 6.53, 542.12 tok/s, 2.24×, 1× train cost (TLT default) | Tbl 7; §6.5 | P |
| 55 | SD methods | TLT defaults to Eagle for the cost / convergence-difficulty tradeoff | §4.1; §6.5 | P |
| 56 | OSD | Qwen2.5-7B target, H100, TP=2, BS=1. Qwen2.5-0.5B: Original 5.95/363.87 → Trained 6.68/408.51 → +OSD 6.89/421.35 | Tbl 8 | P |
| 57 | OSD | Eagle: Original 1.57/130.34 → Trained 6.53/542.12 → +OSD 6.77/562.04 | Tbl 8 | P |
| 58 | Case study | 128 requests, Qwen-32B, H100 (TP=4); SD starts when remaining < 32; overall 2.44× over no-SD baseline | Fig 14 | P |
| 59 | Checkpointing | Selective async checkpointing: Vanilla 893 ms → Async 280 ms (3.2×) → Selective Async 97 ms (9.2× vs vanilla) | Fig 17(a); §6.4 ("9.2×") | P |
| 60 | Sequence packing | 13.3 → 29.6 samples/sec (2.2×) vs vanilla batching | Fig 17(b); §6.4 ("2.2×") | P |
| 61 | Overhead | Stage-transition overhead < 1% of RL-step duration | §6.4 (1) | P |
| 62 | Overhead | SD-switch reprefill completes within 3 seconds | §6.4 (2) | P |
| 63 | Overhead | Training the drafter every 10 steps suffices (not every step) | §6.4 (3) | P |
| 64 | Drafter accuracy | Fig 15: top-3 next-token accuracy rises after warm-up, dips on target update, recovers within a few iterations (x-axis to ≈14K drafter steps) | Fig 15 | P |
| 65 | Testbed | 8× NVIDIA DGX H100 = 64 GPUs; per server 8 H100, 2× Intel Xeon Platinum 8480C (112 cores), 2 TB memory | §6.1 Testbed | P |
| 66 | Testbed | 900 GB/s NVLink intra-node; 400 Gb/s Mellanox InfiniBand inter-node; separate A100 cluster also used | §6.1 Testbed | P |
| 67 | RL settings | GRPO; temperature 0.9; max generation length 32K tokens; TP ∈ {2,4,8}; Adam optimizer; BF16 (rollout + inference) | §6.1 RL Settings | P |
| 68 | Models | Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-32B, Llama-3.3-70B-Instruct | §6.1 Models | P |
| 69 | Data | RL: Eurus-2-RL subset (math LaTeX answers + code test cases); drafter warm-up: OpenThoughts2-1M subset | §6.1 Datasets | P |
| 70 | Baselines | Open-R1 (TRL+vLLM+DeepSpeed, separate placement); VeRL (SOTA, colocation via GPU time-sharing); TLT-Base (vanilla SD, model-free drafter) | §6.1 Baselines | P |
| 71 | Stack | Built on VeRL + Ray; ZeroMQ coordinator; EAGLE training pipelines + FSDP2; async ckpt via PyTorch DCP; SGLang rollout backend; model-free drafter = Lookahead [72] | §6.1 Implementation | P |
| 72 | Worker states | Worker cycles BUSY → IDLE → TRAINING; worker = one rollout instance (e.g., TP=8 ⇒ 1 worker = 8 GPUs); centralized ZeroMQ coordinator (rank 0); leader-election to start training | §4.2 Worker Coordinator | P |
| 73 | DataBuffer | One-step-offset sampling: persists buffer across steps, prioritizes previous step's long sequences; partial+buffered ≈ full-set training | §4.2 Spot Training with DataBuffer | P |
| 74 | Lossless | Reward curves of VeRL and TLT overlap closely for Qwen2.5-7B and -32B (≈steps 0–110) | Fig 12 | P |
| 75 | Async positioning | Async/continuous-batching RL "mixes multiple policy versions," biasing gradients, drifting on-policy → off-policy | §7 "Can We Break RL Synchronization?" | P |
| 76 | Async positioning | TLT preserves on-policy/lossless training; can combine with limited async RL "without introducing additional loss or policy staleness" | §7 | P |
| 77 | Async positioning | Cited async/long-tail RL systems: AReaL [12], StreamRL [75], APRIL [77], RollPacker [14] (RollPacker = tail batching; limited because trajectories vary in length) | §7; §8 | P |
| 78 | Generality | Compatible RL algorithms: GRPO, RLOO, DAPO, REINFORCE, REINFORCE++ | §7 "Can TLT be applied…" | P |
| 79 | Design principles | Four §3 principles (verbatim names): "Lossless Guarantee", "Interference-Free", "Automatic and Simple", "Generalizable and Scalable" | §3 "Design Principles & Goals" | P |
| **D1** | Long-tail math | Batch completion time $T_{\text{batch}}=\max_i (L_i/\text{throughput}_i)$ | Dashboard Model Math §1 | D |
| **D2** | Long-tail math | Pareto tail CDF $F(x)=1-(x_{\min}/x)^{\alpha}$, $\alpha>1$ | Dashboard §1 | D |
| **D3** | Long-tail math | Expected max (closed form) $\mathbb{E}[M_B]=x_{\min}\,B^{1/\alpha}\,\frac{\alpha}{\alpha-1}$ | Dashboard §1 | D |
| **D4** | Long-tail math | Worked example (verbatim): "For α = 2, B = 128: the ratio is ≈ 5.7× — the expected slowest request takes nearly 6× longer than the median." (ratio is **vs median**) | Dashboard §1 | D |
| **D4b** | Long-tail math | ⚠️ The stated ≈5.7× does **not** follow from D3: D3 at α=2,B=128 gives $\mathbb{E}[M_B]\approx22.6\,x_{\min}$ vs Pareto median $x_{\min}\sqrt2$ ⇒ ≈16×, not 5.7×. Dashboard internal inconsistency; reported as the dashboard's figure only. | Author check of D3/D4 | S |
| **D5** | Long-tail math | **[Assumption]** Pareto tail used for a tractable closed form; paper Fig 1a is empirical (log-normal-shaped) | Dashboard §1 (assumption tag) | D |
| **D6** | SD Markov | Expected accepted tokens $\mathbb{E}[n]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ ($\alpha$=per-token accept prob, $\gamma$=draft depth) | Dashboard §2 | D |
| **D7** | SD Markov | Speedup $=\frac{\mathbb{E}[n]}{1+\gamma/c}$, $c=T_{\text{verify}}/T_{\text{draft}}$ | Dashboard §2 | D |
| **D8** | SD Markov | **[Assumption]** Uniform per-token accept rate $\alpha$; real rates decline with position (cf. paper Fig 16) | Dashboard §2 (assumption tag) | D |
| **D9** | Threshold | Benefit decay $b(k)=b_0 e^{-k/\tau}$ ($k$=active requests); activation when aggregate benefit $k\,b(k)\ge O_{\text{SD}}$ (switch overhead) | Dashboard §3 | D |
| **D10** | Threshold | **[Assumption]** Exponential benefit decay with $\tau=16$, chosen to match Table 4's pattern; not stated in paper | Dashboard §3 (assumption tag) | D |
| **D11** | Convergence | Distillation objective $\mathcal{L}(\theta_t)=\mathbb{E}_{x\sim\pi_t}[-\log p_\theta(x)]$ | Dashboard §4 | D |
| **D12** | Convergence | Condition: if $\eta_d I_t>\delta_\pi$ then drafter accept rate $\alpha_t$ non-decreasing in expectation ($\eta_d$=drafter step gain, $I_t$=Fisher info, $\delta_\pi$=policy drift/step) | Dashboard §4 | D |
| **D13** | Convergence | KL decomposition (verbatim incl. higher-order term) $D_{\text{KL}}(\pi_{t+1}\Vert p_{\theta_{t+1}})\le D_{\text{KL}}(\pi_t\Vert p_{\theta_t})-\eta_d I_t+\delta_\pi+O(\eta_d^2)$ | Dashboard §4 | D |
| **D14** | Convergence | **[Assumption]** Fisher info ≈ constant; $\alpha\approx\exp(-D_{\text{KL}})$ holds locally | Dashboard §4 (assumption tag) | D |
| **D15** | Variant A | Variance-aware threshold $k^*(t)=k_0(1+\lambda\,V(t)/V(0))$, $V(t)=\mathrm{Var}(L_i\mid L_i>t)$ | Dashboard §5 (Variant A) | D |
| **D16** | Variant B | Multi-drafter ensemble: per-bucket $\mathbb{E}[n_j]=\frac{1-\alpha_j^{\gamma+1}}{1-\alpha_j}$, aggregate $\mathbb{E}[n_{\text{ens}}]=\sum_{j=1}^{K} w_j\,\mathbb{E}[n_j]$, $w_j=|B_j|/B$ | Dashboard §5 (Variant B) | D |
| **D17** | Variant C | Curriculum order $\sigma^*=\mathrm{argsort}_i(-\hat L_i/c_i(s))$, $c_i$=drafter confidence | Dashboard §5 (Variant C) | D |
| **D18** | Open problems | (1) sample complexity of online distillation vs drift; (2) optimal depth/verify under non-uniform accept ($\alpha_k=\alpha_1\beta^{k-1}$); (3) learnable threshold via secondary RL; (4) cross-task drafter transfer bounds | Dashboard §6 | D |
| **D19** | BibTeX | `hu2026taminglongtailefficientreasoning`, eprint 2511.16665, archivePrefix arXiv, primaryClass cs.LG, year 2026 | Dashboard (citation block) | D |

## Recorded discrepancies / non-uses
- **Dashboard §3 "1.08× at BS=32" (now verified verbatim):** the dashboard states "Table 4 shows SD speedup at BS=1 is 3.66× but drops to 1.08× at BS=32." This is **factually inconsistent with paper Tbl 4**, which shows BS=32 = 1.70× (verify-64) up to 2.48× (verify-16) — **no 1.08× appears anywhere in Tbl 4**. The dashboard's "3.66× at BS=1" is actually the **Fig 13** peak (depth-12/verify-64), not a Tbl 4 entry. **The dashboard's 1.08× is NOT used;** batch-size collapse is anchored to paper Tbl 4. The dashboard's *threshold model* (D9/D10) is still presented as legitimate assumption-tagged scaffolding. (See `provenance_check.md`.)
- **Fig 13 full heatmap:** only the two reported extrema (peak speedup 3.66×; max accept length 9.78) are used as anchors; intermediate heatmap cells are not transcribed to avoid OCR error.
- **Fig 11 Open-R1 per-model bars:** recorded only as an approximate range (row 8); not all bar labels are individually legible.
- **Dashboard ≈5.7× (row D4):** reproduced as the dashboard's own illustrative figure; its exact normalization (vs median/mean) is the dashboard's and is **not** independently endorsed here.

## Code grounding ([Code] column — Phase 1)

Mechanistic claims grounded against `mit-han-lab/fastrl` @ `main` HEAD `bce3df7` (on-disk shallow clone; SGLang vendored at `third-party/sglang/`). All paths relative to repo root. Tag **[Code]** = confirmed in source; verdicts from `phase1_code_grounding.md`. These complement (do not replace) the [P]/[D]/[S] rows above.

| # | Mechanism / default | Value or formula (code) | `file:line` | Verdict | Tag |
|---|---|---|---|---|---|
| C1 | Spot Trainer (non-blocking drafter training) | `EagleBackgroundTrainer` + `RolloutDrafterManager`; preemptible asyncio task on freed GPUs | `verl/workers/drafter/eagle_background_trainer.py:35`; `verl/workers/rollout/sglang_rollout/worker_manager.py:411` | CONFIRMED | [Code] |
| C2 | Worker state machine | 5 states `IDLE/GENERATING/RELEASED/TRAINING/COMPLETED` (paper's BUSY/IDLE/TRAINING = GENERATING/RELEASED/TRAINING) | `worker_manager.py:54-61` | CONFIRMED (refines paper) | [Code] |
| C3 | Coordinator transport | ZeroMQ; rank-0 coordinator; REP/PUB (coord) + REQ/SUB (workers) | `worker_manager.py:91-107,441,314-335` | CONFIRMED | [Code] |
| C4 | Promotion threshold | `min_workers_for_training=1` (configurable) | `verl/trainer/config/fastrl_trainer.yaml:25`; `worker_manager.py:436` | CONFIRMED | [Code] |
| C5 | Drafter training cadence | `training_interval_steps=10` (shipped YAML); code `.get(...,1)` fallback | `fastrl_trainer.yaml:20`; `worker_manager.py:455` | CONFIRMED (reconciles paper "~10 steps") | [Code] |
| C6 | DataBuffer one-step-offset | collect when `(step+1)%interval==0`; train draws last `buffer_steps=2`; `max_size=10000` | `worker_manager.py:490-493`; `eagle_background_trainer.py:68,276` | CONFIRMED | [Code] |
| C7 | DataBuffer long-seq prioritization | `random.sample` over last 2 steps + 512-tok response window — **no explicit length ranking** | `eagle_background_trainer.py:299-356` | PARTIAL (softer than paper) | [Code] |
| C8 | Selective async ckpt | `dcp.async_save` + drop frozen `{model.embed_tokens.weight, lm_head.weight}`; cadence `(step//100)` | `eagle_background_trainer.py:83,105-148,596` | CONFIRMED (mechanism); 9.2× is [Paper Fig 17a] only | [Code] |
| C9 | Drafter loss | `SmoothL1Loss` + soft-target CE; `vloss_weight=ploss_weight=0.5` | `eagle_background_trainer.py:73,558-578` | CONFIRMED (refines "L1+CE") | [Code] |
| C10 | Single-layer drafter | `num_hidden_layers=1`; embed frozen both paths; LM head frozen (EAGLE) / trainable (EAGLE-3) | `eagle-train/eagle_trainer.py:190,255-258`; `eagle3_trainer.py:475` | CONFIRMED (path-dependent LM head) | [Code] |
| C11 | Drafter warm-up data | OpenThoughts2-1M (sampled ~250k); warmup_steps 2000 | `eagle-train/dataset/create_mixed_dataset.py:23-24`; `eagle_trainer.py:36` | CONFIRMED | [Code] |
| C12 | Elastic SD threshold | `bs_threshold=32`; enable on `batch_size ≤ 32` after **10 consecutive checks** (`adaptive_spec_warmup_checks=10`) | `fastrl_trainer.yaml:9`; `examples/grpo_7B.sh:28`; `eagle_worker.py:361,364,196` | CONFIRMED (warmup hysteresis paper-omitted) | [Code] |
| C13 | BEG-MAB reward (exact) | `a = sum(accept_len)/bs + 1`; `reward = stable_accept_length·bs/total_time` | `eagle_worker.py:1234,1240` | CONFIRMED (exact match to Alg 1) | [Code] |
| C14 | BEG-MAB defaults | `ε=0.1`, `window=1000`, `bs_threshold=[1,2,5,21]`, arm `steps_topk_drafttokens`, OOM `draft_tokens·bs ≤ 2048` | `eagle_mab.py:210,220,111,36,442` | CONFIRMED | [Code] |
| C15 | Bucketed CUDAGraph | spec buckets `[1-8]+[10-32/2]+[40-64/4]+[72-257/8]+[272-max/16]`; merge by `draft_tokens`; draft=DECODE, target=TARGET_VERIFY | `server_args.py:789-796`; `cuda_graph_runner.py:169-207`; `eagle_mab.py:237-245` | CONFIRMED | [Code] |
| C16 | Tree drafting | topK branch ×`speculative_num_steps`; select `num_draft_tokens-1` via `torch.topk`; `verify_tree_greedy` | `eagle_worker.py:748-821`; `eagle_utils.py:15-34` | CONFIRMED | [Code] |
| C17 | Model-free n-gram drafter | `NGRAMWorker`/`NgramCache`; `max_match_window=12`, `branch_length=18`, BFS 1–10, capacity 10M | `ngram_worker.py:119-138`; `server_args.py:342-347` | CONFIRMED | [Code] |
| C18 | RL run config | GRPO `n=8`; KL `low_var_kl` coef 0.001 (not in reward); temp 0.9; max-gen 32K; TP 4(7B)/8(32B); FSDP2; Ulysses SP 4/8 | `examples/grpo_7B.sh`; `examples/grpo_32B_multi_nodes.sh` | CONFIRMED (confirms + extends §6.1) | [Code] |
| C19 | HF drafters | `mit-han-lab/Qwen2.5-7B-Eagle-RL`, `-32B-Eagle-RL` | `examples/bench_sd.sh:2`; `grpo_7B.sh:12`; `grpo_32B_multi_nodes.sh:24` | CONFIRMED | [Code] |
| C20 | Online-training gate (caveat) | `enable_drafter_training=false`, `collect_hidden_states_from_sgl=false` (default OFF) | `fastrl_trainer.yaml:19,26` | CONFIRMED (reproducibility caveat) | [Code] |
