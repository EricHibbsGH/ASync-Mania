# Phase 1 — Code Grounding

**Repo:** `mit-han-lab/fastrl` @ `main` HEAD `bce3df7` (shallow clone on disk at `…/fastrl`). Apache-2.0. SGLang is vendored in-tree (`third-party/sglang/`).

**Method.** Each mechanism was located by direct `Grep`/`Read` and cross-checked by an independent 11-agent code-grounding workflow (12 agents incl. synthesizer; 347 tool calls). **Verdict tally: 11/11 CONFIRMED, 0 PARTIAL / NOT_FOUND / DIVERGES at the mechanism level** (within-CONFIRMED caveats listed in §C). Claims I *personally opened the file for* are marked ✔︎-read; the rest were read by sub-agents that returned verbatim `file:line` snippets (re-readable on disk).

Canonical path note: all SGLang paths are under `third-party/sglang/python/sglang/srt/...` (one agent printed a `fastrl/third-party/...` prefix — same tree under the repo root).

---

## A. Mechanism → code table

| # | Mechanism (paper) | Verdict | Key `file:line` | Symbol | Read |
|---|---|---|---|---|---|
| 1 | **Spot Trainer** — non-blocking drafter training on partial responses, preemptible spot tasks on idle GPUs | CONFIRMED | `verl/workers/drafter/eagle_background_trainer.py:35`; `verl/workers/rollout/sglang_rollout/worker_manager.py:411` | `EagleBackgroundTrainer`, `RolloutDrafterManager` | ✔︎ |
| 2 | **Worker Coordinator** — state machine, ZeroMQ rank-0, idle-threshold promotion, leader-election, graceful shutdown | CONFIRMED | `worker_manager.py:54-61, 91-107, 245-283, 204-230` | `CentralCoordinator`, `WorkerClient`, `WorkerState`, `CoordinatorEvent` | ✔︎ |
| 3 | **DataBuffer** — cache hidden states+tokens, one-step-offset cross-step sampling | CONFIRMED | `verl/utils/data_buffer.py:85-95`; `worker_manager.py:490-493` | `DataBuffer.get_data_from_last_n_steps`, `should_collect_data_this_step` | ✔︎ |
| 4 | **Selective async checkpointing** — background thread, trainable-only dump, PyTorch DCP | CONFIRMED | `eagle_background_trainer.py:83, 105-117, 119-148, 596` | `_get_trainable_state_dict`, `_save_checkpoint_async`, `dcp.async_save` | ✔︎ |
| 5 | **Sequence packing** — varlen concat + attention mask | CONFIRMED | drafter side `eagle_background_trainer.py:410-417`; RL side `verl/models/mcore/util.py:43-46,68` + `flash_attn_varlen_func` | `preprocess_packed_seqs`, `cu_seqlens` | ✔︎ (drafter) / agent (mcore) |
| 6 | **Single-layer drafter** — 1 trainable decoder layer, frozen embed+LM head, EAGLE-style, OpenThoughts2 warm-up | CONFIRMED | `eagle-train/eagle_trainer.py:190, 255-258`; `verl/workers/fsdp_workers.py:559-569` | `num_hidden_layers=1`, `requires_grad=False` | agent (eagle-train) / ✔︎ (fsdp freeze + trainer frozen-set) |
| 7 | **Adaptive SD Manager + elastic activation** — enable SD when running requests ≤ threshold (default 32) | CONFIRMED | `eagle_worker.py:361,364,383`; `verl/trainer/config/fastrl_trainer.yaml:9`; `examples/grpo_7B.sh:28` | `should_enable_sd`, `adaptive_spec_threshold` | ✔︎ |
| 8 | **Bucketed CUDAGraph capture** — bucketed batch sizes, disaggregated target/draft, merged captures | CONFIRMED | `cuda_graph_runner.py:169-207`; `server_args.py:789-796`; `eagle_mab.py:237-245` | `get_batch_sizes_to_capture`, `EAGLEDraftCudaGraphRunner` | agent |
| 9 | **BEG-MAB tuner** — ε-greedy over `(steps,topK,draft_tokens)`, bucketed, median-reward, exact reward formula | CONFIRMED | `eagle_mab.py:197-292`; reward in `eagle_worker.py:1234,1240` | `BucketedEpsilonGreedyMAB`, `record_mab_strategy_metrics` | ✔︎ |
| 10 | **Tree drafting** — topK branching for Draft_Depth, select Tokens_to_Verify | CONFIRMED | `eagle_worker.py:748-821`; `spec_utils.py:388-435`; `eagle_utils.py:15-34` | `select_top_k_tokens`, `organize_draft_results`, `verify_tree_greedy` | agent |
| 11 | **Model-free drafter** — n-gram retrieval fallback (Lookahead-style, ref [72]) | CONFIRMED | `ngram_worker.py:119-138`; `cpp_ngram/ngram_cache.py:48-59`; `spec_info.py:9-14` | `NGRAMWorker`, `NgramCache`, `SpeculativeAlgorithm.NGRAM` | agent |

---

## B. Highest-value corrections (code defaults the paper rounds or omits)

These are the reviewer-grade findings — concrete constants/formulas that sharpen, and in a few places correct, the paper.

1. **BEG-MAB reward formula — EXACT, code-grounded** [Code: `eagle_worker.py:1234,1240`] ✔︎: `accept_length_avg = sum(accept_length_per_req_cpu)/batch_size + 1`; `reward = stable_accept_length * batch_size / total_time` (time from CUDA events, seconds). Matches Algorithm 1 exactly; refinement: in multi-strategy mode `stable_accept_length` is the **median** accept length across batch-size groups, not the raw per-step average (`eagle_mab.py:493-521`).
2. **BEG-MAB defaults the paper leaves implicit** [Code: `eagle_mab.py:210,220,111`] ✔︎: **ε = 0.1**, **window = 1000**, bucket thresholds **`bs_threshold=[1,2,5,21]`**; arm encoding `steps_topk_draft_tokens`; OOM guard = `draft_tokens × batch_size ≤ 2048` (`eagle_mab.py:442`). Note: four MAB variants exist (`EG/UCB1/PREDEFINED/BEG`); Algorithm 1 = the BEG variant.
3. **Elastic-SD threshold = 32, with a hysteresis the paper omits** [Code: `eagle_worker.py:361,364`; `fastrl_trainer.yaml:9`; `grpo_7B.sh:28`] ✔︎: SD activates on `batch_size ≤ 32` only after **10 consecutive qualifying checks** (`adaptive_spec_warmup_checks=10`); the counter resets above threshold; EXTEND batches always run.
4. **Drafter training cadence — reconciles the paper's "~10 steps"** [Code: `fastrl_trainer.yaml:20`] ✔︎: the shipped config sets `training_interval_steps=10` (matches the paper's "every ~10 steps is sufficient"). The code `.get(...,1)` fallback in `worker_manager.py:455`/`ray_trainer.py` defaults to 1 — **YAML wins at runtime**.
5. **Worker state machine is 5 states, not 3** [Code: `worker_manager.py:54-61`] ✔︎: `IDLE / GENERATING / RELEASED / TRAINING / COMPLETED`. The paper's "BUSY/IDLE/TRAINING" maps as **BUSY=GENERATING**, **IDLE(memory released)=RELEASED**.
6. **Promotion threshold default `min_workers_for_training=1`** [Code: `fastrl_trainer.yaml:25`] ✔︎ — a single released worker suffices to start drafter training; `max_steps=200` cap per training activation (`worker_manager.py:723`).
7. **Selective checkpoint = exact frozen set** [Code: `eagle_background_trainer.py:83`] ✔︎: `{"model.embed_tokens.weight","lm_head.weight"}` excluded; drafter ckpt cadence `(step//100) > _last_ckpt_step` (every 100 trainer steps).
8. **Drafter loss = SmoothL1 + soft-target CE, 0.5/0.5** [Code: `eagle_background_trainer.py:73,558-578`] ✔︎ — refines the paper's "L1+CE": regression term is `SmoothL1Loss` (Huber), distillation term is `−Σ target_p·log_softmax(logits)`, weights `vloss_weight=ploss_weight=0.5`. Drafter training window capped at **512 tokens** (`:356`); `effective_batch_size=min(bs,4)` (`:287`); `batch_size_per_gpu=2`, `data_buffer_max_size=10000`, `buffer_steps=2` (one-step offset).
9. **GRPO/run defaults the paper omits** [Code: `examples/grpo_7B.sh`, `grpo_32B_multi_nodes.sh`] ✔︎: GRPO group **n=8**; train batch 64 prompts (7B)/32·nodes (32B), mini-batch 4/8; **KL via `low_var_kl` loss, coef 0.001, NOT in reward** (`use_kl_in_reward=False`); LR 1e-6; FSDP2; Ulysses SP 4 (7B)/8 (32B); TP 4 (7B)/8 (32B); gpu_mem_util 0.4/0.55. Confirms paper: temp 0.9, max-gen 32K, sglang, Eurus-2-RL, GRPO, `mit-han-lab/Qwen2.5-{7B,32B}-Eagle-RL`.
10. **CUDAGraph bucket schedules (exact)** [Code: `server_args.py:789-796`]: spec = `[1-8] + [10-32 by 2] + [40-64 by 4] + [72-257 by 8] + [272-max by 16]`; merge key = `draft_tokens` group; draft graphs use `ForwardMode.DECODE`, target uses `TARGET_VERIFY`.
11. **N-gram drafter defaults** [Code: `server_args.py:342-347`]: `max_match_window_size=12`, `branch_length=18`, BFS breadth 1–10, cache capacity 10M tokens.

---

## C. Caveats & within-CONFIRMED divergences (the honest ledger)

- **9.2× checkpoint claim — mechanism present, multiplier is an end-to-end measurement, not a param-count ratio.** The code confirms async `dcp.async_save` + trainable-only dump [`eagle_background_trainer.py:119-148`], i.e. the *mechanism* for both the 3.2× (async) and the further 9.2× (selective) steps. But the exact 9.2× is **not reproducible from code**; it is the paper's Fig 17a measurement. One sub-agent estimated frozen embed+lm_head at ~3.7% of a 7B model and concluded selectivity can't yield ~3×; **that framing is wrong** — the checkpoint is of the *drafter* (single decoder layer + tied embed/LM head), for which the (frozen) embed+LM head dominate the parameter count, so excluding them plausibly explains the extra ~2.9× beyond async. **Net: report 893→280→97 ms / 3.2×→9.2× as [Paper Fig 17a]; the selectivity+async mechanism is [Code]; the magnitude is not independently verified.**
- **"Frozen LM head" is path-dependent.** On the EAGLE path the LM head is frozen (`eagle_trainer.py:257-258`); on the **EAGLE-3** path the LM head is **trainable** for the draft-vocabulary subset (`eagle3_trainer.py:475`). The paper's default is EAGLE, so "frozen embed+LM head" holds for the default but not universally. Also `tie_word_embeddings=false` in the Eagle config — the embed/LM head are *frozen copies loaded from the target*, not weight-tied in the HF sense; "reused & frozen" is the precise phrasing.
- **Throughput multipliers (2.2× packing, end-to-end speedups) have no in-code assertion** — the mechanisms are standard and present; the numbers are empirical paper claims, not code constants.
- **Everything is opt-in / off by default.** `enable_drafter_training=false` and `collect_hidden_states_from_sgl=false` in `fastrl_trainer.yaml` gate the entire Spot-Trainer/DataBuffer/online-training pipeline; the shipped example scripts run adaptive SD with a **pre-trained** `*-Eagle-RL` drafter and do **not** flip on online spot-training. The machinery is fully wired and present; reproducing the "drafter trained for free during RL" headline requires setting these flags. **This is the single most important reproducibility caveat.**
- **DataBuffer "prioritize long sequences" is softer in code than in the paper.** The buffer is step-tagged FIFO (`deque`), training draws via `random.sample` over the last `buffer_steps=2` steps plus a response-centered 512-token window (`eagle_background_trainer.py:299-356`). The cross-step "one-step-offset" is confirmed; an explicit *length-ranked* prioritization of prior long sequences is **not** present as written — flag as PARTIAL on that sub-claim.

---

## D. Net verdict

Every mechanism the paper claims is implemented and locatable in the released code; the BEG-MAB reward formula and the elastic threshold match the paper exactly. The corrections in §B (and caveats in §C) are now folded into `paper.md` (Phase 6) with `[Code: path]` tags, and into `data_table.md`'s new code-grounding section. No fabricated paths: every `file:line` above is on disk in the shallow clone.
