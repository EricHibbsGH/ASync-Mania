# Overlapping, Not Approximating: Asynchronous-Style Speedup in Reasoning-RL Training and the Lossless Design of TLT

*A survey-and-analysis paper anchored on* Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter *(Hu, Yang, Guo, Yao, Lin, Gu, Cai, Gan, Klimovic, Han; ASPLOS '26; arXiv:2511.16665v2), with analytical framing drawn from the interactive companion dashboard by Eric Hibbs, and mechanism claims grounded in the released code (github.com/mit-han-lab/fastrl).*

> Provenance conventions (four axes). Every quantitative or mechanistic claim is tagged: [Paper §/Fig/Tbl] = stated in the TLT paper (*measured*); [Dashboard §N] = derived/assumed by the dashboard author (Hibbs) in "Model Math" (*modeled, assumption-based, NOT a paper result*); [Code: path] = confirmed against the released repo (*implemented*; `mit-han-lab/fastrl` @ commit `bce3df7`); [Author synthesis] = my own derivation or framing (*synthesized*). Direct quotes are verbatim, < 15 words, at most one per source per claim; everything else is paraphrase. "Lossless" throughout means token-level distributional equivalence of rollouts, speculative-decoding verification preserves the target model's exact sampling distribution; it is not a claim about convergence or final accuracy. Throughput numbers are per-step token throughput at matched reward curves, not wall-clock-to-convergence.

---

## Abstract

Reinforcement-learning training of reasoning LLMs is throttled by a *long-tail* rollout problem: a handful of very long generations dominate each step while most workers sit idle. TLT attacks this not by approximating the algorithm but by overlapping otherwise-idle work in time. Its two pillars, an *Adaptive Drafter* trained opportunistically on idle GPUs during the tail, and an *Adaptive Rollout Engine* that turns on speculative decoding (SD) exactly when the batch shrinks into SD's sweet spot, convert wasted tail bubbles into useful compute while keeping training token-level lossless and strictly on-policy. The paper reports "over 1.7×" end-to-end speedup over state-of-the-art systems [Paper Abstract], with the figure-level range given as 1.7–2.1× over VeRL [Paper Fig 11], a geomean of 1.86× on H100 and 1.70× on A100, and a peak of 2.12× on Qwen-32B/H100 [Paper Fig 11, Tbl 3]. Rollout alone is ≈85% of step time [Paper §1, Fig 1a]. This paper frames TLT around its *asynchrony/concurrency* mechanisms, non-blocking drafter training, preemptible opportunistic scheduling, selective asynchronous checkpointing, and tail-phase adaptive SD, pairs each with its measured speedup *and its implementation in the released code*, develops a mathematical model of where the gains come from and why they saturate (using the dashboard's order-statistic, Markov-acceptance, optimal-stopping, and moving-target-convergence derivations, each with its assumptions flagged), and contrasts TLT's *lossless* synchrony with the off-policy drift risked by fully asynchronous RL systems. The thesis: you can capture async-style utilization gains without paying the correctness tax of async RL.

---

## 1. Introduction

### 1.1 The long-tail bottleneck

A GRPO-style RL step has three stages, rollout (generate candidate responses), inference (score them and compute a KL penalty), and training (update the policy). The first stage dominates: the paper measures the rollout phase at approximately 85% of total step time [Paper §1 item (1), Fig 1a]. Worse, rollout latency is governed by the *slowest* response in the batch, and response lengths are heavy-tailed: most are short, a few hit the configured maximum. The production evidence is stark. A ByteDance trace [Paper Fig 2] of Qwen2.5-32B on 128 H20-96GB GPUs ran 11 days for only 385 steps, averaging ≈40 min/step, with evaluation every 5 steps costing ≈20 min each, under a customized max length of 20,480 tokens [Paper §1 item (3), Fig 2]. Because the batch finishes only when its longest member finishes, the gap between the maximum and the 75th percentile is a direct measure of wasted GPU-time, an "under-utilized zone" visible across the entire run [Paper Fig 2]. (A schematic of this distribution and the under-utilized zone is in [Figures](#figures), `fig1_longtail_SCHEMATIC`, illustrative only, since the raw per-request distribution is not published as numbers.)

### 1.2 Why asynchronous *overlap*, not approximation

There are two ways to make a tail-bound workload faster. One is to approximate: drop the stragglers, accept stale gradients, or mix policy versions, the route taken by fully asynchronous RL. These trade correctness for utilization. The other is to overlap: keep the algorithm exact, but fill the idle time the tail creates with useful, independent work. TLT takes the second route. Its design rests on two insights stated by the paper: rollout bubbles can be exploited because resources freed as short sequences finish can be repurposed at no extra cost, and drafter training can be decoupled from rollout completion so that its computation overlaps ongoing rollouts rather than following them [Paper §1, §4.2]. Speculative decoding is the ideal payload because it is "Mathematically Lossless" [Paper §2.2], the accepted token distribution is identical to the target's, and it is most effective precisely in the tail, where effective batch sizes are small and decoding is memory-bound. The result is *async-style utilization* (idle GPUs doing work) with *synchronous-grade correctness* (the RL update is unchanged).

### 1.3 Contributions of this paper

This is a survey-and-analysis paper, not a new system. Its contributions are: (i) a reframing of TLT around the concurrency mechanisms that produce its speedups, each paired with its measured number and its location in the released code (§4, grounded by a full mechanism→code audit); (ii) a consolidated mathematical model of the speedup and its saturation, built from the dashboard's derivations with every assumption flagged and connected to the paper's tables (§3, §5); (iii) a synthesis of TLT's evaluation (§6); (iv) an analysis of why TLT stays losslessly synchronous where async-RL systems risk off-policy drift (§7); (v) an Implementation & Reproducibility account and a Threats-to-Validity / Assumptions ledger (§8–§9); and (vi) a four-way-attributed treatment (paper / dashboard / code / synthesis) of every claim.

---

## 2. Background

### 2.1 Reasoning RL and GRPO's three stages

Reasoning LLMs improve through *test-time scaling*: longer chains of thought and self-reflection raise accuracy, and these behaviors are largely acquired during RL [Paper §2.1]. The paper uses Group Relative Policy Optimization (GRPO), which replaces PPO's learned critic with a group-relative baseline, cutting overhead [Paper §2.1]. A single GRPO step [Paper Fig 4] is:

1. Rollout, the target model generates multiple responses per prompt (autoregressive decoding; the long-tail stage). The released runs use a group size of n = 8 responses/prompt [Code: examples/grpo_7B.sh].
2. Inference, responses are prefilled by both the target and a frozen reference model to compute a KL-divergence penalty; a rule-based reward (e.g., answer correctness) is computed. The code applies KL as a loss term (`low_var_kl`, coefficient 0.001), not folded into the reward [Code: examples/grpo_7B.sh].
3. Training, a loss combining reward and KL updates the target weights.

The asymmetry matters: rollout is sequential token-by-token generation (latency-bound, tail-dominated), while inference and training are batched prefills/backward passes. TLT inserts its opportunistic work into stage 1's tail and reuses stage 2's hidden states.

### 2.2 Speculative decoding: lossless, and a memory→compute shift

In speculative decoding a small draft model proposes a short run of tokens, and the large target verifies them all in one parallel forward pass, accepting the longest correct prefix plus one corrected token [Paper §2.2, Fig 5]. Two properties make SD ideal for RL. First, it is token-level lossless: the verification step preserves the target's exact next-token sampling distribution, so the accepted sequence is distributed identically to ordinary autoregressive sampling from the target (this is a *distributional* guarantee on rollouts, not a statement about convergence). Second, it changes the *arithmetic intensity*: autoregressive decoding at small batch is memory-bound (one token's weights streamed per step); verifying γ speculative tokens at once is compute-bound, better saturating the GPU. The paper's roofline analysis shows SD reaching peak TFLOPS at far smaller batch sizes than vanilla decoding [Paper Fig 5c], i.e., SD's benefit is largest exactly where the tail lives (few in-flight requests). The paper summarizes the mechanism as shifting from "memory-bound towards compute-bound" [Paper §2.2].

Three challenges block naïve SD in RL [Paper §2.2/§2 "Challenges"]: C1, the evolving target model makes a fixed drafter go stale; C2, drafter training cost; C3, fluctuating batch sizes, since SD that helps at batch 1 can hurt or OOM at large batch. TLT's two components answer C1/C2 (Adaptive Drafter) and C3 (Adaptive Rollout Engine).

### 2.3 What "asynchronous" means here

This is the crux of the framing, and the word is overloaded. In async-RL systems (e.g., AReaL), "asynchronous" means *algorithmic* asynchrony: the policy is updated from rollouts generated by *earlier* policy versions, so training and generation run on different model versions concurrently, fast, but off-policy. In TLT, "asynchronous" means *systems-level* overlap with *no* algorithmic change: the policy update still consumes only on-policy rollouts of the current version; what runs concurrently is the *drafter's* training and the *checkpoint's* save, neither of which touches the policy gradient. TLT's first design principle is an explicit "Lossless Guarantee" that forbids asynchronous staleness in the RL update itself [Paper §3]. So the entire speedup must come from overlapping *auxiliary* work, never from desynchronizing the *core* work, the distinction this paper builds on.

---

## 3. The Long-Tail, Formally

The paper's evidence for the tail is empirical: the response-length histogram with a max length of 30K for Qwen-7B and DeepSeek-R1-7B [Paper Fig 1a], and the ByteDance percentile trace [Paper Fig 2]. The dashboard supplies a *motivating* closed-form analysis on top of this. It is useful intuition but rests on an assumption the paper does not make, so it is flagged throughout.

### 3.1 Batch completion as an order statistic

Let a batch hold $B$ responses with token-lengths $L_1,\dots,L_B$. If response $i$ decodes at throughput $\text{throughput}_i$, the batch finishes only when its slowest member does:
$$T_{\text{batch}} = \max_{i=1}^{B}\frac{L_i}{\text{throughput}_i}.\qquad\text{[Dashboard §1]}$$
The bottleneck is therefore the maximum, a classical order statistic, the formal version of "a few very long responses dominate execution time" [Paper Abstract].

### 3.2 Expected maximum under a Pareto tail (with the assumption flagged)

The dashboard models the tail as Pareto, $P(L>x) = (x_{\min}/x)^{\alpha}$ with shape $\alpha>1$, i.e. CDF $F(x)=1-(x_{\min}/x)^{\alpha}$ [Dashboard §1]. This is an [Assumption]: the dashboard explicitly notes the paper's Fig 1a is empirical (log-normal-shaped) and that Pareto is chosen only for a tractable closed form [Dashboard §1 assumption tag]. With that caveat, the dashboard states the expected maximum as
$$\mathbb{E}[M_B] = x_{\min}\cdot B^{1/\alpha}\cdot\frac{\alpha}{\alpha-1}.\qquad\text{[Dashboard §1]}$$

Step-by-step derivation [Author synthesis]. Let $M_B=\max_i L_i$ with the $L_i$ i.i.d. Pareto. Then $P(M_B\le x)=F(x)^B$. The exact expectation is
$$\mathbb{E}[M_B] = x_{\min}\,\frac{\Gamma(B+1)\,\Gamma\!\left(1-\tfrac{1}{\alpha}\right)}{\Gamma\!\left(B+1-\tfrac{1}{\alpha}\right)},\qquad \alpha>1.$$
Using the ratio asymptotic $\Gamma(B+1)/\Gamma(B+1-\tfrac{1}{\alpha})\sim B^{1/\alpha}$ for large $B$ (Stirling), this becomes $\mathbb{E}[M_B]\approx x_{\min}B^{1/\alpha}\Gamma(1-\tfrac1\alpha)$. The dashboard's form replaces the tail constant $\Gamma(1-\tfrac1\alpha)$ with $\tfrac{\alpha}{\alpha-1}$, the Pareto mean $\mathbb{E}[L]=x_{\min}\tfrac{\alpha}{\alpha-1}$. So the dashboard's closed form is best read as *"scale the mean by the order-statistic growth factor $B^{1/\alpha}$."*

What it means physically. The straggler grows as $B^{1/\alpha}$: the heavier the tail (smaller $\alpha$), the faster the worst case blows up as you widen the batch, and the deeper the rollout bubble, and, read in reverse, the more freed capacity becomes available for opportunistic work.

Numerical recomputation and a flagged error [Author synthesis]. I recomputed the α=2, B=128 case (exact Γ formula + 2×10⁵-sample Monte-Carlo): $\mathbb{E}[M_B]\approx\mathbf{20.07\,x_{\min}}$ (exact) / 20.14 (MC); the dashboard's own closed form gives 22.63 (it mildly over-states, since $\tfrac{\alpha}{\alpha-1}=2$ exceeds $\Gamma(\tfrac12)=\sqrt\pi\approx1.77$). The Pareto median is $x_{\min}\sqrt2$, so the true expected-max/median ratio is ≈14× (exact/MC), or 16× under the dashboard's formula. The dashboard's worked example states "≈5.7× vs median" [Dashboard §1], this is numerically wrong: $5.7\approx\sqrt{128}/2=B^{1/\alpha}/\alpha$, a mis-normalization (it drops the mean factor and mis-divides). The qualitative point (the straggler is many-fold the median) holds, but the *specific* multiplier is the dashboard's and should not be cited as a derived result; the correct figure is ≈14×.

### 3.3 From the math back to the paper's measurement

The order-statistic view predicts that under-utilization is structural, not incidental, which is exactly what the paper observes empirically and persistently across 385 steps [Paper Fig 2], and what motivates harvesting the bubble rather than merely shortening it.

---

## 4. Mechanisms of Async Speedup

Each subsection names a concurrency mechanism, pairs it with its measured effect [Paper], and grounds it in the released code [Code] (full audit: `phase1_code_grounding.md`; all 11 mechanisms located and confirmed). Collectively they convert the tail bubble (§3) into the end-to-end gains of §6.

### 4.1 Non-blocking Spot Trainer + DataBuffer (overlapping drafter training with rollout)

The core asynchrony is decoupling drafter training from rollout completion. Vanilla RL runs rollout → inference → training in sequence; a naïve SpecRL bolts drafter training on as another sequential step [Paper Fig 8a,b]. The Spot Trainer, class `EagleBackgroundTrainer` [Code: verl/workers/drafter/eagle_background_trainer.py:35], instead trains the drafter on *partial* responses as soon as enough short sequences finish, executing "in a non-blocking manner" during the tail [Paper §4.2, Fig 8c].

Training only on early finishers would bias the drafter away from long sequences, so TLT adds the DataBuffer [Code: verl/utils/data_buffer.py]: it caches hidden states and tokens from the inference stage and uses one-step-offset sampling, confirmed in code as collecting data the step *before* training (`should_collect_data_this_step` fires at `(step+1) % interval == 0`) and drawing the training batch from the last `buffer_steps = 2` RL steps [Code: worker_manager.py:490-493; eagle_background_trainer.py:276]. *Provenance nuance:* the paper describes prioritizing the previous step's long sequences; the released code samples the last two steps via `random.sample` over a response-centered 512-token window, the cross-step offset is confirmed, but an explicit *length-ranked* prioritization is not present as written [Code; see §9]. The paper reports that partial current-step data combined with buffered sequences attains performance comparable to full-set training [Paper §4.2]. The hidden states are *reused* from the inference phase, so drafter training adds no extra prefill; the target is "over 20×" larger than the draft model [Paper §4.1].

*Measured effect:* this mechanism makes the drafter a "free byproduct" [Paper Abstract], its training cost is hidden inside the bubble.

### 4.2 Worker Coordinator state machine (preemptible opportunistic scheduling)

The bubble must be harvested without disturbing rollout. A centralized Worker Coordinator (`RolloutDrafterManager` + `CentralCoordinator`, rank-0, ZeroMQ; REP/PUB on the coordinator, REQ/SUB on workers) tracks each worker, one rollout instance, e.g. a TP=8 worker spans 8 GPUs [Paper §4.2; Code: worker_manager.py:91-107,441]. The paper summarizes the lifecycle as BUSY → IDLE → TRAINING [Paper §4.2]; the released code implements a five-state machine, `IDLE / GENERATING / RELEASED / TRAINING / COMPLETED`, where the paper's "BUSY" = `GENERATING` and "IDLE (memory released)" = `RELEASED` [Code: worker_manager.py:54-61]. Once enough workers release, the coordinator promotes them to training (default threshold `min_workers_for_training = 1`) and, on rollout completion, broadcasts `STOP_TRAINING`/`BATCH_COMPLETE` for graceful shutdown [Code: worker_manager.py:245-283,204-230; fastrl_trainer.yaml:25]. This realizes TLT's "Interference-Free" principle [Paper §3], drafter work runs as low-priority "spot tasks" [Paper §4.2]. The shipped drafter-training cadence is `training_interval_steps = 10`, matching the paper's "every ~10 steps is sufficient" [Paper §6.4; Code: fastrl_trainer.yaml:20].

*Measured effect:* stage-transition overhead is <1% of the RL step [Paper §6.4].

### 4.3 Selective asynchronous checkpointing (overlapping the save with training)

Preemptibility implies frequent checkpoints, which would themselves stall training. TLT offloads the save to a background thread (`dcp.async_save`, PyTorch DCP) and dumps only the drafter's trainable parameters, excluding the frozen set `{model.embed_tokens.weight, lm_head.weight}` [Code: eagle_background_trainer.py:83,105-148]. The paper's measured ladder [Paper Fig 17a]:

- Vanilla synchronous checkpoint: 893 ms → Async (background thread): 280 ms (3.2×) → Selective async (background + trainable-only): 97 ms (9.2× vs vanilla) [Paper Fig 17a, §6.4].

*Reading the 9.2× honestly:* this is the paper's end-to-end measurement, not a parameter-count ratio. The 3.2× comes from overlapping I/O; the further step to 9.2× comes from selectivity, and because the object being checkpointed is the single-layer drafter (where the frozen, tied embedding + LM head dominate the parameter count), excluding them is a large reduction. The async + selective *mechanism* is code-confirmed; the exact magnitude is the paper's [Paper Fig 17a].

### 4.4 Drafter architecture and sequence packing (cheap payload, dense compute)

The drafter is a single trainable transformer decoder layer (`config.num_hidden_layers = 1`) that reuses the target's frozen Embedding and LM Head (`requires_grad = False`) [Paper §4.1, Fig 7; Code: eagle-train/eagle_trainer.py:190,255-258]. Only that one layer updates, ≈ 1/layer_num of target parameters [Paper §4.1]. The paper's contrast: Qwen2.5-32B has 64 layers vs Qwen2.5-0.5B's 24; a 64× parameter cut still leaves latency dominated by sequential layers, so TLT's single-layer drafter, at roughly the parameter count of a Qwen2.5-0.5B drafter, is 2.4× faster [Paper §4.1]. *Code refinements:* the drafter loss is `SmoothL1Loss` (Huber) on hidden states + a soft-target cross-entropy on logits, weighted 0.5/0.5, a sharper statement of the paper's "L1 + CE" [Code: eagle_background_trainer.py:73,558-578]; and the "frozen LM head" holds on the default EAGLE path but not on the EAGLE-3 path, where the LM head is trainable for the draft vocabulary [Code: eagle3_trainer.py:475; see §9].

Within the preemptible slots, sequence packing concatenates variable-length sequences (attention-masked, no padding) to keep the GPU dense [Paper §4.2; Code: eagle_background_trainer.py:410-417; verl/models/mcore/util.py]. *Measured effect:* training throughput rises 13.3 → 29.6 samples/s (2.2×) over padded batching [Paper Fig 17b]. (See [Figures](#figures): `fig17_spot_trainer`.)

### 4.5 Adaptive SD + Bucketed CUDAGraph + BEG-MAB (tail-phase, OOM-safe, self-tuning)

SD must be turned on only where it helps. The Adaptive SD Manager activates SD only when the remaining request count drops below a threshold (default 32) [Paper §5.1; Code: examples/grpo_7B.sh:28 `speculative.bs_threshold=32`]. The code adds a hysteresis the paper omits: SD enables only after the batch stays `≤ 32` for 10 consecutive checks (`adaptive_spec_warmup_checks`), and the counter resets above threshold [Code: eagle_worker.py:361,364,196]. Tree-based drafting explores the topK branches for `Draft_Depth` steps and verifies the `Tokens_to_Verify` highest-confidence tokens in parallel [Paper §5.1, Fig 9; Code: eagle_worker.py:748-821].

Bucketed CUDAGraph Capture groups batch sizes into buckets, disaggregates target vs draft graph capture, and merges identical configurations [Paper §5.1; Code: server_args.py:789-796]. *Measured effect* [Paper Tbl 5, Llama-3-8B, TP=4]: Single strategy 7.81 GB; naïve multi-strategy 30.39 GB; Bucketed 10.69 GB, a 2.8× reduction vs naïve at near single-strategy cost.

Strategy selection is automated by the BEG-MAB tuner (`BucketedEpsilonGreedyMAB`): each arm is `(Draft_Depth, topK, Tokens_to_Verify)`, bucketed by `Tokens_to_Verify`, exploiting the median reward over a sliding window [Paper §5.2, Alg 1; Code: eagle_mab.py:197-292]. The exact reward is code-confirmed: `a = sum(accept_len)/batch_size + 1`, `reward = a · batch_size / elapsed_time` [Code: eagle_worker.py:1234,1240], matching Algorithm 1. Code-level defaults the paper omits: ε = 0.1, window = 1000, bucket thresholds [1,2,5,21], and an OOM guard `draft_tokens × batch_size ≤ 2048` [Code: eagle_mab.py:210,220,442]. For early steps, a model-free n-gram drafter (Lookahead-style) is the fallback, with defaults `max_match_window = 12`, `branch_length = 18` [Paper §5.3; Code: ngram_worker.py; server_args.py:342-347].

*Measured effect (case study)* [Paper Fig 14]: 128 requests on Qwen-32B/H100, SD switches on below 32 remaining requests and delivers an overall 2.44× over the no-SD baseline.

---

## 5. A Mathematical Model of the Speedup

Every closed form here is the dashboard author's [Dashboard §2–§4], *modeled, assumption-based*; the paper's numbers are the empirical anchors, and `phase3_accuracy.md` records the numerical re-derivations. These models are qualitative, not predictive fits to the paper's tables.

### 5.1 Speculative decoding as a truncated geometric process

Assume the drafter proposes $\gamma$ tokens and each is independently accepted with probability $\alpha$. This uniform per-token accept rate is an [Assumption], real rates decline with position (paper Fig 16) [Dashboard §2]. Then
$$\mathbb{E}[n] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}.\qquad\text{[Dashboard §2]}$$

Derivation [Author synthesis]. With $A$ the accepted-prefix length, $P(A\ge k)=\alpha^{k}$; the per-position law is $P(A=k)=\alpha^k(1-\alpha)$ for $k<\gamma$ and $P(A=\gamma)=\alpha^\gamma$. Tokens committed $=A+1$, so $\mathbb{E}[n]=\sum_{k=0}^{\gamma}\alpha^k$; the geometric sum gives $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. ∎ Verified: $\alpha\to1\Rightarrow\gamma+1$ (computed 9.000 at γ=8); $\gamma\to\infty\Rightarrow\frac1{1-\alpha}$ (cap 5.0 at α=0.8). This tapering matches the paper's accept-length saturation [Paper Fig 13a], max 9.78 at depth 16/verify 128.
Violation: Fig 16 shows α decaying with depth, so the uniform-α premise does not hold in practice, the model is *qualitative*.

### 5.2 The speedup, and why there is an optimal depth

With $c=T_{\text{verify}}/T_{\text{draft}}$:
$$\text{speedup}=\frac{\mathbb{E}[n]}{1+\gamma/c}.\qquad\text{[Dashboard §2]}$$
Reading [Author synthesis]. Numerator saturates in $\gamma$ (§5.1); denominator grows linearly ⇒ an inverted-U with interior optimum. Verified: $\arg\max_\gamma$ speedup = 3/5/7 for $c$ = 4/8/16, and the optimum grows with $c$, qualitatively matching the paper's grid peak of 3.66× at depth 12/verify 64 [Paper Fig 13b]. The companion topK sweep is nearly flat, accept length 8.29–8.67, speedup 3.47×–3.65× across topK 4–16 [Paper Tbl 1], so TLT fixes topK and tunes the rest. *Note on conditions:* the SD grids (Fig 13/Tbl 1) are measured at temperature 0 for stable accept-length readings; the operative accept rates during training (temp 0.9) are those in Tbl 6/Fig 16. (See [Figures](#figures): `fig13_sd_grid_heatmaps`, `dashboard_markov_surface`.)

### 5.3 The adaptive threshold as optimal stopping (why SD collapses at large batch)

The dashboard models per-request SD benefit at $k$ active requests as exponential decay and asks when aggregate benefit clears the switch cost $O_{\text{SD}}$:
$$b(k)=b_0\,e^{-k/\tau},\qquad\text{activate when } k\cdot b(k)\ge O_{\text{SD}}.\qquad\text{[Dashboard §3]}$$
$\tau=16$ is an [Assumption] fit to Table 4, not in the paper [Dashboard §3].
Reading [Author synthesis]. $g(k)=b_0 k e^{-k/\tau}$ peaks at $k=\tau$ then decays, so beyond some $k$ aggregate benefit no longer clears the overhead, an activation threshold. *Implementation reality:* the deployed rule is not this smooth model but a hard `batch_size ≤ 32` step with a 10-check warmup [Code: eagle_worker.py:361,196]; the $b(k)$ model only explains the *direction* of the collapse.
Empirical anchor, and a dashboard error. Paper Tbl 4 (Qwen-32B, TP=4, depth 10, topK 8) shows, at verify-64, BS=1 → 3.62× falling to BS=32 → 1.70× (2.48× at verify-16). The dashboard's §3 text instead claims SD "drops to 1.08× at BS=32" while citing Table 4 [Dashboard §3], but no 1.08× appears in Table 4, and its companion "3.66× at BS=1" is the Fig 13 grid peak, not a Tbl 4 entry [Paper Tbl 4; Fig 13b]. The error is the dashboard's; only the paper's values are used here. (See [Figures](#figures): `tbl4_batch_erosion`, `dashboard_threshold_curve`.)

### 5.4 The drafter as a moving target (convergence under policy drift)

The drafter minimizes $\mathcal{L}(\theta_t)=\mathbb{E}_{x\sim\pi_t}[-\log p_\theta(x)]$ [Dashboard §4]. Using $\alpha\approx e^{-D_{\text{KL}}}$ locally and a per-step KL recursion (with $\eta_d$ the drafter step gain, $I_t$ Fisher information, $\delta_\pi$ policy drift):
$$D_{\text{KL}}(\pi_{t+1}\Vert p_{\theta_{t+1}})\le D_{\text{KL}}(\pi_t\Vert p_{\theta_t})-\eta_d I_t+\delta_\pi+O(\eta_d^2),$$
the accept rate is non-decreasing in expectation iff $\eta_d I_t>\delta_\pi$ [Dashboard §4]. Both rest on an [Assumption] (Fisher ≈ constant; local KL–accept relation).
What it means [Author synthesis]. The drafter keeps up iff its per-step learning outpaces policy drift, the skeleton of three paper observations: the dip-then-recover after each target update [Paper Fig 15]; co-training lifting RL accept length 4.59→6.53 and downstream 3.76→5.15 [Paper Tbl 6]; and sustained accept rate at distant token indices [Paper Fig 16]. Arithmetically $(6.53-4.59)/4.59=42.3\%$ [Author synthesis], matching the dashboard's "+42%"; the dashboard reads Fig 16 as ≈32% (adaptive) vs ≈6% (vanilla) at index 8, the dashboard's reading, not a printed paper value [Dashboard §4]. (See [Figures](#figures): `tbl6_adaptive_drafter`, `fig16_accept_decay_SCHEMATIC`.)

---

## 6. Evaluation Synthesis

### 6.1 Setup (so the numbers carry their conditions)

TLT is built on VeRL + Ray, a ZeroMQ coordinator, EAGLE training pipelines under FSDP2, asynchronous checkpointing via PyTorch DCP, and SGLang as the rollout backend; the model-free fallback is Lookahead [Paper §6.1; all code-confirmed, Phase 1]. Testbed: 8× DGX H100 = 64 GPUs (per node: 8 H100, 2× Intel Xeon Platinum 8480C, 2 TB RAM; 900 GB/s NVLink, 400 Gb/s InfiniBand), plus a separate A100 cluster [Paper §6.1]. Models: Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-32B, Llama-3.3-70B-Instruct [Paper §6.1]. Algorithm: GRPO, temperature 0.9, max generation 32K, TP ∈ {2,4,8}, Adam, BF16 [Paper §6.1; Code confirms n=8, KL `low_var_kl`/0.001, LR 1e-6]. Data: Eurus-2-RL (math+code) for RL; an OpenThoughts2-1M subset to warm up the drafter [Paper §6.1]. Baselines: Open-R1, VeRL (SOTA), TLT-Base [Paper §6.1]. Metric: per-step token throughput averaged over three steps after warm-up, i.e., gains are *step-throughput at matched reward curves* (Fig 12), not a claim of fewer steps to convergence [Paper §6.1].

### 6.2 Consolidated results

| Result | Condition | Number | Source |
|---|---|---|---|
| End-to-end vs VeRL (headline) | over SOTA | >1.7× (abstract) / 1.7–2.1× (figure) | [Paper Abstract; Fig 11] |
| End-to-end geomean | H100 / A100 | 1.86× / 1.70× | [Paper Fig 11] |
| End-to-end peak | Qwen-32B, H100, 8 nodes | 2.12× | [Paper Fig 11; Tbl 3] |
| TLT vs VeRL, H100 | Qwen-7B / DS-7B / Qwen-32B / Llama-70B | 1.76 / 1.71 / 2.12 / 1.86× | [Paper Fig 11] |
| TLT-Base vs VeRL, H100 | same models | 1.42 / 1.38 / 1.54 / 1.31× | [Paper Fig 11] |
| Rollout share of step |, | ≈85% | [Paper §1; Fig 1a] |
| SD speedup, peak grid | Qwen-32B, BS=1, depth 12/verify 64, temp 0 | 3.66× | [Paper Fig 13b] |
| SD accept length, max | depth 16/verify 128, temp 0 | 9.78 | [Paper Fig 13a] |
| topK insensitivity | Qwen-32B, depth 12/verify 64 | 3.47–3.65× (topK 4–16) | [Paper Tbl 1] |
| Batch sensitivity (verify 64) | Qwen-32B, depth 10 | 3.62× (BS=1) → 1.70× (BS=32) | [Paper Tbl 4] |
| GPU diversity (rollout, BS1/TP1) | Qwen-7B micro-benchmark | B200 2.33× · H100 2.61× · A100 2.79× · RTX5090 2.91× · RTX4090 2.87× · RTX3090 3.22× | [Paper Tbl 2] |
| Cluster scale (Qwen-7B) | 1/2/4/8 nodes | 1.21 / 1.45 / 1.62 / 1.76× | [Paper Tbl 3] |
| Cluster scale (Qwen-32B) | 1/2/4/8 nodes | OOM / OOM / 1.83 / 2.12× | [Paper Tbl 3] |
| Bucketed CUDAGraph | Llama-3-8B, TP=4, 4 strategies | 7.81 → 30.39 → 10.69 GB (2.8× vs naïve) | [Paper Tbl 5] |
| Selective async ckpt | drafter checkpoint | 893 → 280 → 97 ms (9.2×) | [Paper Fig 17a] |
| Sequence packing |, | 13.3 → 29.6 samples/s (2.2×) | [Paper Fig 17b] |
| Adaptive drafter accept len | RL / downstream, Base→R | 4.59→6.53 / 3.76→5.15 | [Paper Tbl 6] |
| SD method (Eagle, default) | Qwen-7B, TP=2, BS=1 | 6.53 acc, 542.12 tok/s, 2.24×, 1× train cost | [Paper Tbl 7] |
| Case study (adaptive SD) | 128 req, Qwen-32B, H100 | 2.44× over no-SD | [Paper Fig 14] |
| Overheads |, | stage-transition <1%, SD reprefill <3 s, drafter every 10 steps | [Paper §6.4] |

### 6.3 GPU diversity and scalability

SD's relative gain grows on more I/O-bound hardware: 2.33× on B200 up to 3.22× on RTX 3090 [Paper Tbl 2]. *Caveat:* this is a BS=1, TP=1 rollout micro-benchmark, not full-scale training, it isolates the memory→compute shift (slower memory subsystems gain more), and is not representative of end-to-end training throughput. End-to-end gains grow with cluster and model size: Qwen-7B 1.21×→1.76× (1→8 nodes); Qwen-32B is OOM at 1–2 nodes and reaches 2.12× at 8 nodes [Paper Tbl 3], larger jobs have longer tails and bigger bubbles to harvest. (See [Figures](#figures): `tbl2_gpu_diversity`, `tbl3_cluster_scaling`, `fig11_endtoend_speedup`.)

### 6.4 Drafter quality and SD-method choice

Among SD methods in TLT (Qwen-7B, TP=2, BS=1): Base/No-SD 242.11 tok/s (1.00×); Eagle 542.12 tok/s, 2.24×, 1× cost; HASS 553.91 tok/s, 2.29×, 3× cost; Eagle-3 617.42 tok/s, 2.55×, 7× cost [Paper Tbl 7]. TLT defaults to Eagle because higher-accept rivals require multi-pass "training-time test" that inflates per-step cost and convergence difficulty, unaffordable in a small rollout bubble [Paper §6.5]. OSD-style training further lifts a Qwen2.5-0.5B drafter (5.95→6.68→6.89) and an Eagle drafter (1.57→6.53→6.77) [Paper Tbl 8].

### 6.5 Overheads and losslessness

Overheads are small: stage-transition <1%; SD-switch reprefill <3 s; drafter trained only every ≈10 steps [Paper §6.4]. Most important for the thesis: VeRL and TLT reward curves overlap closely for Qwen2.5-7B and -32B [Paper Fig 12], direct evidence the overlap-based speedup leaves learning dynamics unchanged. The gains are *utilization* (step throughput), not approximation; the paper does not claim, and this survey does not assert, a reduction in the *number of steps* to a target reward.

---

## 7. Why Not Fully Async?

### 7.1 The tempting next step, and its hidden cost

If idle GPUs can train the drafter, why not also update the *policy* asynchronously from partial rollouts during the long decode? The paper addresses this in "Can We Break RL Synchronization?" [Paper §7]. Asynchronous RL with continuous batching "mixes multiple policy versions" within a single rollout [Paper §7], biasing gradient estimation and drifting on-policy → off-policy. TLT's SD approach composes with *limited* async RL "without introducing additional loss or policy staleness" [Paper §7].

### 7.2 The lossless/off-policy contrast in the model

The §5.4 condition $\eta_d I_t>\delta_\pi$ makes the tradeoff legible [Author synthesis]. TLT tolerates *drafter* staleness because the drafter affects only SD's *speed*, never the verified output distribution, a stale drafter lowers the accept rate but cannot corrupt a token (SD is token-level lossless, §2.2). Async RL instead tolerates *policy* staleness, which enters the gradient directly: there the analogue of $\delta_\pi$ is not a tracking error to be corrected but a bias injected into the update. TLT confines all tolerated staleness to the *auxiliary* path and keeps the *core* path exact.

### 7.3 Positioning vs async / long-tail RL systems

The paper situates TLT against systems that relax synchronization differently, and it is worth keeping the categories distinct (Phase 2 reconciliation):

- Off-policy / staleness-tolerant: AReaL (large-scale asynchronous RL) [arXiv:2505.24298] and APRIL (active partial rollouts that over-provision then recycle incomplete rollouts across steps) [arXiv:2509.18521] mix or carry over policy versions.
- Disaggregation: StreamRL (disaggregated streaming generation) [arXiv:2504.15930].
- Synchronous long-tail scheduling: RollPacker (tail batching into long/short rounds) is itself synchronous [arXiv:2509.21009], it is a scheduling method, *not* an async-RL system; the paper notes its tail batching is limited because trajectories for identical queries vary widely in length [Paper §8].

These accelerate training but (for the off-policy cases) risk degrading quality by admitting stale responses or truncating rollouts. TLT's differentiator is that it preserves the original RL algorithm and is *orthogonal*, it can layer on top of limited-async RL rather than compete with it [Paper §7, §8]. The same overlap discipline generalizes across RL algorithms: GRPO, RLOO, DAPO, REINFORCE, REINFORCE++ share the workflow TLT accelerates [Paper §7].

---

## 8. Implementation & Reproducibility

Grounded in the on-disk repository (`mit-han-lab/fastrl`, Apache-2.0, `main` @ `bce3df7`; full audit in `phase1_code_grounding.md`).

Stack & layout. Top-level `verl/` (281 .py, RL trainer, workers, coordinator, DataBuffer), `eagle-train/` (12 .py, drafter trainers + warm-up), `third-party/sglang/` (vendored in-tree, 1287 .py, adaptive SD, BEG-MAB, tree drafting, n-gram), `examples/` (`bench_sd.sh`, `grpo_7B.sh`, `grpo_32B_multi_nodes.sh`). Adaptive-SD work tracks SGLang PR #4732 (paper ref [59]). Released drafters: `mit-han-lab/Qwen2.5-7B-Eagle-RL`, `-32B-Eagle-RL` [Code: bench_sd.sh:2; grpo_*.sh].

Reproduced run configs. 7B: TP=4, FSDP2, Ulysses SP=4, GRPO n=8, batch 64 prompts/mini 4, max-gen 32K, temp 0.9, KL `low_var_kl`/0.001, `speculative.bs_threshold=32`, 8 GPUs/1 node [Code: grpo_7B.sh]. 32B: TP=8, SP=8, 8 nodes × 8 GPUs = 64, max-prompt 2048 [Code: grpo_32B_multi_nodes.sh], consistent with the Tbl 3 8-node / 2.12× and OOM rows.

Key reproducibility caveat (the most important finding). The online "drafter trained for free during RL" pipeline is off by default in the shipped config: `enable_drafter_training=false` and `collect_hidden_states_from_sgl=false` [Code: fastrl_trainer.yaml:19,26]. The example scripts demonstrate adaptive SD with a pre-trained `*-Eagle-RL` drafter; reproducing the headline "free byproduct" claim requires enabling these flags. The machinery (`EagleBackgroundTrainer` + coordinator + DataBuffer) is fully present and wired.

Reception & follow-on work (search-sourced, June 2026, post-cutoff). TLT was covered by MIT News, *"New method could increase LLM training efficiency"* (2026-02-26), syndicated across MIT EECS / Schwarzman College / Climate Portal and Mirage News; the article frames the speedup in popular-press terms (~70–210%), a loose rendering of the paper's precise 1.7–2.1×, not adopted here. Near-simultaneous concurrent work, *"Beat the long tail: Distribution-Aware Speculative Decoding for RL Training"* (arXiv:2511.13841), addresses the same problem from a distribution-aware angle; it is concurrent, not a TLT result.

---

## 9. Threats to Validity / Assumptions Ledger

Dashboard assumptions (modeled, not paper-validated):
1. Pareto tail (§3.2), paper Fig 1a is empirical/log-normal; the closed form is for tractability. The dashboard's "5.7× vs median" is numerically wrong (true ≈14×, §3.2).
2. Uniform per-token accept rate α (§5.1), contradicted by Fig 16 (α decays with depth); the Markov model is qualitative.
3. Exponential benefit decay, τ=16 (§5.3), a fit, not in the paper; the deployed rule is a hard `≤32` step with 10-check warmup.
4. Fisher information ≈ constant; local α≈e^(−D_KL) (§5.4), convergence framing only.
 Plus the dashboard's §3 batch numbers ("1.08× at BS=32") are inconsistent with Tbl 4 and are not used.

Schematic / approximate figures: `fig1_longtail_*` and `fig16_*` are SCHEMATIC (only stated anchors are real); `fig13` interior cells and `fig11` Open-R1 bars are figure-read/approximate; `dashboard_*` surfaces are modeled, not measured (figures/manifest.md).

Code vs paper divergences (within-CONFIRMED): worker states are 5 not 3 (§4.2); "frozen LM head" is EAGLE-only, not EAGLE-3 (§4.4); DataBuffer uses random sampling rather than explicit long-sequence prioritization (§4.1); the 9.2× checkpoint figure is an end-to-end measurement, not a parameter ratio (§4.3).

Scope of the speedup claims: all speedups are per-step token throughput (or per-decode SD speedup), measured at matched reward curves [Paper Fig 12], not wall-clock-to-convergence or accuracy gains. GPU-diversity numbers are a BS=1/TP=1 micro-benchmark. End-to-end gains are workload-dependent: they grow with tail severity, model size, and cluster size, and shrink when tails are short. The drafter is "adequate for SD speedup," not claimed SOTA as a standalone model; the BEG-MAB carries ε=0.1 exploration overhead and assumes near-stationarity within its 1000-step window.

---

## 10. Extensions and Open Problems

The dashboard proposes three forward-looking variants, [Dashboard §5 / Author synthesis], design sketches *not* evaluated by the paper, inheriting the §5 assumptions.

A. Variance-aware thresholding. $k^\*(t)=k_0(1+\lambda\,V(t)/V(0))$, $V(t)=\mathrm{Var}(L_i\mid L_i>t)$ [Dashboard §5A]. *Rationale [Author synthesis]:* a high-variance residual tail warrants enabling SD earlier, generalizing the static "<32" rule into a workload-adaptive one.

B. Multi-drafter ensembling. $\mathbb{E}[n_j]=\frac{1-\alpha_j^{\gamma+1}}{1-\alpha_j}$, $\mathbb{E}[n_{\text{ens}}]=\sum_{j} w_j\mathbb{E}[n_j]$, $w_j=|B_j|/B$ [Dashboard §5B]. *Rationale:* different batch regimes have different accept profiles; a weighted ensemble could dominate any single drafter, at the memory cost Bucketed CUDAGraph (§4.5) is built to contain.

C. Curriculum-ordered rollouts. $\sigma^\*=\mathrm{argsort}_i(-\hat L_i/c_i(s))$, $c_i$ = drafter confidence [Dashboard §5C]. *Rationale:* starting the likely-longest, hardest-to-draft sequences first could shorten the critical path.

Open problems [Dashboard §6]. (1) sample complexity of online distillation vs the policy drift rate (the quantitative §5.4 condition); (2) optimal depth/verify under *non-uniform* accept ($\alpha_k=\alpha_1\beta^{k-1}$, relaxing §5.1); (3) a learnable threshold trained by a secondary RL loop (replacing fixed τ/"<32"); (4) cross-task drafter transfer bounds. The paper independently notes compatible regimes, uniformly-long responses, multi-turn tool-calling RL, fixed-target serving/edge deployment [Paper §7].

---

## 11. Conclusion

TLT's lesson is methodological as much as empirical: the long-tail tax in reasoning RL is best paid back by overlapping concurrent work in time, not by approximating the algorithm. By training a one-layer drafter opportunistically on idle GPUs (non-blocking Spot Trainer + DataBuffer), harvesting bubbles with a preemptible coordinator, hiding checkpoint I/O (9.2×) and padding waste (2.2×), and switching on token-level-lossless speculative decoding only in the memory-bound tail (2.44× in the case study, peak 3.66× per-decode), TLT delivers >1.7× end-to-end (1.7–2.1× over VeRL; geomean 1.86×/1.70× on H100/A100; peak 2.12×) [Paper Abstract, Fig 11, Tbl 3] while leaving reward curves indistinguishable from the synchronous baseline [Paper Fig 12]. The mathematical model explains the magnitude and saturation of these gains *qualitatively*, with the caveats that those closed forms are the dashboard author's assumption-based scaffolding (one of which, the 5.7× Pareto example, is numerically wrong) and that the released code refines several paper details (5-state machine, EAGLE-only frozen LM head, online-training off by default). The standing result: async-style utilization and synchronous-grade correctness are not mutually exclusive, a design point the fully-asynchronous RL literature has yet to occupy.

---

## Figures

All figures are regenerated by `make_figures.py` (vector `.pdf`+`.svg`, colorblind-safe); provenance and exact-vs-schematic status are in `figures/manifest.md`.

![End-to-end speedup](figures/fig11_endtoend_speedup.svg)
*Figure 1.* End-to-end RL training speedup vs VeRL=1.0, H100 + A100. [Paper Fig 11] (TLT/TLT-Base/VeRL exact; Open-R1 bars figure-read/approximate; some A100 Open-R1 cells illegible → n/a).

![Batch erosion](figures/tbl4_batch_erosion.svg)
*Figure 2.* SD speedup vs Tokens_to_Verify across batch sizes; the collapse to 1.70× motivates the elastic threshold. [Paper Tbl 4] (exact).

![GPU diversity](figures/tbl2_gpu_diversity.svg)
*Figure 3.* Rollout SD speedup across GPUs (BS=1, TP=1 micro-benchmark). [Paper Tbl 2] (exact).

![Cluster scaling](figures/tbl3_cluster_scaling.svg)
*Figure 4.* End-to-end speedup vs cluster size; 32B OOM at 1–2 nodes. [Paper Tbl 3] (exact).

![Spot-Trainer optimizations](figures/fig17_spot_trainer.svg)
*Figure 5.* Selective async checkpointing (893→280→97 ms, 9.2×) and sequence packing (13.3→29.6 samples/s, 2.2×). [Paper Fig 17] (exact).

![Adaptive drafter accept length](figures/tbl6_adaptive_drafter.svg)
*Figure 6.* Accept length Target-Base→Target-R (+42% RL). [Paper Tbl 6] (exact).

![SD grid heatmaps](figures/fig13_sd_grid_heatmaps.svg)
*Figure 7.* Depth×Verify grids for accept length and speedup; boxed extrema (9.78, 3.66×) verified. [Paper Fig 13] (boxed extrema exact; interior cells figure-read/approximate).

![Accept-rate decay](figures/fig16_accept_decay_SCHEMATIC.svg)
*Figure 8.* Accept rate vs draft-token index, adaptive vs vanilla. [Paper Fig 16, SCHEMATIC; endpoints ~32%/~6% from Dashboard].

![Markov speedup surface](figures/dashboard_markov_surface.svg)
*Figure 9.* Modeled SD speedup over (α, γ). [Dashboard §2, assumption-based (uniform α); NOT measured].

![Threshold curve](figures/dashboard_threshold_curve.svg)
*Figure 10.* Aggregate SD benefit k·b(k), peak at k=τ=16. [Dashboard §3, assumption-based; NOT measured].

![Long-tail schematic](figures/fig1_longtail_SCHEMATIC.svg)
*Figure 11.* Long-tail length distribution + under-utilized zone. [Paper Fig 1a/Fig 2, SCHEMATIC, not to scale; raw data not published as numbers].

---

## References

*Primary and secondary sources, then external works actually cited. External bibliographic details follow the TLT paper's reference list; arXiv/venue strings verified in Phase 5 (`phase5_citations.md`).*

[Primary, Paper] Q. Hu, S. Yang, J. Guo, X. Yao, Y. Lin, Y. Gu, H. Cai, C. Gan, A. Klimovic, S. Han. *Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter.* ASPLOS '26, Pittsburgh, PA, Mar 22–26 2026. arXiv:2511.16665v2 (21 Jan 2026). DOI 10.1145/3779212.3790231. Code (Apache-2.0): github.com/mit-han-lab/fastrl.

[Secondary, Dashboard] E. Hibbs. *Taming the Long-Tail, an interactive companion* ("Model Math" §1–6). https://erichibbsgh.github.io/Taming-the-Long-Tail-an-interactive./ (accessed 2026-06-05).

[Code artifact] `mit-han-lab/fastrl`, Apache-2.0, `main` @ commit `bce3df7`; HF drafters `mit-han-lab/Qwen2.5-{7B,32B}-Eagle-RL`.

External works cited:
- Z. Shao et al. *DeepSeekMath* (GRPO). arXiv:2402.03300, 2024.
- DeepSeek-AI et al. *DeepSeek-R1.* arXiv:2501.12948, 2025.
- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- C. Chen et al. *Accelerating LLM Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.
- Y. Li, F. Wei, C. Zhang, H. Zhang. *EAGLE.* ICML 2024. · *EAGLE-2.* EMNLP 2024. · *EAGLE-3.* arXiv:2503.01840, 2025.
- L. Zhang, X. Wang, Y. Huang, R. Xu. *HASS: Learning Harmonized Representations for Speculative Sampling.* ICLR 2025.
- T. Cai et al. *Medusa.* arXiv:2401.10774, 2024. · Z. Ankner et al. *Hydra.* COLM 2024.
- X. Miao et al. *SpecInfer.* ASPLOS 2024.
- X. Liu et al. *Online Speculative Decoding (OSD).* ICML 2024.
- G. Sheng et al. *HybridFlow (VeRL).* EuroSys 2025.
- Hugging Face. *Open-R1.* 2025.
- W. Fu et al. *AReaL: A Large-Scale Asynchronous RL System for Language Reasoning.* arXiv:2505.24298, 2025.
- Y. Zhong et al. *StreamRL.* arXiv:2504.15930, 2025.
- Y. Zhou et al. *APRIL: Active Partial Rollouts in RL to Tame Long-tail Generation.* arXiv:2509.18521, 2025.
- W. Gao et al. *RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training.* arXiv:2509.21009, 2025.
- Y. Zhong et al. *RLHFuse (Optimizing RLHF with Stage Fusion).* NSDI 2025.
- Z. Mei et al. *ReaLHF.* arXiv:2406.14088, 2024.
- L. Zheng et al. *SGLang.* NeurIPS 2024. · P. Moritz et al. *Ray.* OSDI 2018. · P. Hintjens. *ZeroMQ.* O'Reilly, 2013.
- PyTorch Team. *FSDP2*; *Torch Distributed Checkpoint.* 2025.
- M. M. Krell et al. *Efficient Sequence Packing.* arXiv:2107.02027, 2022.
- Y. Zhao et al. *Lookahead.* KDD 2024.
- Qwen et al. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2025.
- G. Cui et al. *Eurus-2-RL / Process Reinforcement through Implicit Rewards.* arXiv:2502.01456, 2025.
- OpenThoughts. *OpenThoughts2-1M.* 2025. · L. Zheng et al. *MT-Bench.* 2023.
- Q. Yu et al. *DAPO.* arXiv:2503.14476, 2025. · A. Ahmadian et al. *RLOO (Back to Basics).* ACL 2024. · J. Hu et al. *REINFORCE++.* arXiv:2501.03262, 2025.
- D. Kingma, J. Ba. *Adam.* ICLR 2015. · ByteDance. *W&B trace of DAPO Reproduction on verl.* 2025.

Concurrent / reception (search-sourced, post-cutoff, dated):
- MIT News. *New method could increase LLM training efficiency.* news.mit.edu/2026/…-0226 (2026-02-26).
- *Beat the long tail: Distribution-Aware Speculative Decoding for RL Training.* arXiv:2511.13841, 2025 (concurrent work).

### BibTeX for the primary paper

```bibtex
@inproceedings{hu2026tlt,
  title     = {Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter},
  author    = {Hu, Qinghao and Yang, Shang and Guo, Junxian and Yao, Xiaozhe and
               Lin, Yujun and Gu, Yuxian and Cai, Han and Gan, Chuang and
               Klimovic, Ana and Han, Song},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural
               Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26)},
  year      = {2026},
  doi       = {10.1145/3779212.3790231},
  note      = {arXiv:2511.16665v2}
}
```
*(The repo README's BibTeX reads "31th", corrected here to "31st"; its News-line link text uses a non-canonical title, the title above is the canonical PDF title.)*
