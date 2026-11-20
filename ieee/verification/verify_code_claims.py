#!/usr/bin/env python3
"""verify_code_claims.py -- check every code citation in the paper against fastrl@bce3df7.

Usage:
    git clone https://github.com/mit-han-lab/fastrl && cd fastrl && git checkout bce3df7
    python verify_code_claims.py /path/to/fastrl

Each check states the paper's claim, the file and line range cited, and a literal
substring that must appear in that range. Exit status is non-zero on any failure.
"""
import os
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "fastrl"
SG = "third-party/sglang/python/sglang/srt"
ET = "verl/workers/drafter/eagle_background_trainer.py"
WM = "verl/workers/rollout/sglang_rollout/worker_manager.py"
YAML = "verl/trainer/config/fastrl_trainer.yaml"

CLAIMS = [
    # (claim, file, first_line, last_line, required substring)
    ("online drafter training off by default", YAML, 19, 19, "enable_drafter_training: false"),
    ("hidden-state collection off by default", YAML, 26, 26, "collect_hidden_states_from_sgl: false"),
    ("SD batch threshold default 32", YAML, 9, 9, "bs_threshold: 32"),
    ("drafter trained every 10 RL steps", YAML, 20, 20, "training_interval_steps: 10"),
    ("one released worker suffices to start training", YAML, 25, 25, "min_workers_for_training: 1"),
    ("BEG-MAB bucket boundaries", YAML, 17, 17, "[1, 2, 5, 21]"),
    ("five-state worker machine: RELEASED", WM, 54, 61, 'RELEASED = "released"'),
    ("five-state worker machine: COMPLETED", WM, 54, 61, 'COMPLETED = "completed"'),
    ("collection gated on drafter training", WM, 489, 493, "if not self.train_drafter"),
    ("one-step-offset collection", WM, 489, 493, "(self.current_rl_step + 1) % self.training_interval_steps == 0"),
    ("STOP_TRAINING message", WM, 78, 79, 'STOP_TRAINING = "stop_training"'),
    ("Spot Trainer class", ET, 35, 35, "class EagleBackgroundTrainer"),
    ("Smooth-L1 hidden-state loss", ET, 73, 73, "SmoothL1Loss"),
    ("frozen params excluded from checkpoint", ET, 83, 83, '"model.embed_tokens.weight", "lm_head.weight"'),
    ("asynchronous DCP checkpoint", ET, 119, 148, "dcp.async_save"),
    ("buffer draws from last 2 RL steps", ET, 276, 276, "buffer_steps: int = 2"),
    ("uniform random sampling from buffer", ET, 299, 304, "random.sample"),
    ("packing: concatenation without padding", ET, 410, 417, "torch.cat(input_ids_list, dim=0)"),
    ("packing: all-ones attention mask", ET, 410, 417, "torch.ones((1, total_seq_len)"),
    ("training step skipped without hidden states", ET, 473, 479, "if not collect_hidden_states_from_sgl"),
    ("drafter forward call", ET, 504, 509, "attention_mask=batch[\"attention_mask\"]"),
    ("drafter positions default to arange", "verl/workers/drafter/model/qwen2_eagle.py", 88, 89, "position_ids = cache_position.unsqueeze(0)"),
    ("FlashAttention-2 replaced by eager", "verl/workers/drafter/model/qwen2_eagle.py", 193, 197, "config._attn_implementation = 'eager'"),
    ("single-layer drafter", "eagle-train/eagle_trainer.py", 190, 190, "config.num_hidden_layers = 1"),
    ("frozen embedding / LM head (EAGLE)", "eagle-train/eagle_trainer.py", 256, 258, "requires_grad = False"),
    ("EAGLE-3 LM head trainable", "eagle-train/eagle3_trainer.py", 473, 476, "LM head needs to be trained"),
    ("SD warm-up hysteresis = 10 checks", f"{SG}/speculative/eagle_worker.py", 196, 196, "adaptive_spec_warmup_checks = 10"),
    ("SD enabled after consecutive checks", f"{SG}/speculative/eagle_worker.py", 364, 364, "adaptive_spec_consecutive_checks >= self.adaptive_spec_warmup_checks"),
    ("BEG-MAB reward", f"{SG}/speculative/eagle_worker.py", 1234, 1240, "reward = stable_accept_length * batch_size / total_time"),
    ("tree drafting forward", f"{SG}/speculative/eagle_worker.py", 748, 748, "def draft_forward"),
    ("BEG-MAB class", f"{SG}/speculative/eagle_mab.py", 197, 197, "class BucketedEpsilonGreedyMAB"),
    ("BEG-MAB epsilon 0.1", f"{SG}/speculative/eagle_mab.py", 210, 210, "epsilon: float = 0.1"),
    ("sliding window 1000", f"{SG}/speculative/eagle_mab.py", 111, 111, "window_size: int = 1000"),
    ("OOM guard draft_tokens x batch <= 2048", f"{SG}/speculative/eagle_mab.py", 442, 442, "<= 2048"),
    ("n-gram match window 12", f"{SG}/server_args.py", 342, 342, "speculative_ngram_max_match_window_size: int = 12"),
    ("n-gram branch length 18", f"{SG}/server_args.py", 346, 346, "speculative_ngram_branch_length: int = 18"),
    ("bucketed CUDA-graph batch sizes", f"{SG}/server_args.py", 789, 796, "list(range(272, self.cuda_graph_max_bs + 1, 16))"),
    ("GRPO n=8 (7B script)", "examples/grpo_7B.sh", 15, 15, "n_resp_per_prompt=8"),
    ("KL as loss, low_var_kl", "examples/grpo_7B.sh", 53, 53, "kl_loss_type=low_var_kl"),
    ("KL coefficient 0.001", "examples/grpo_7B.sh", 52, 52, "kl_loss_coef=0.001"),
    ("32B run uses 8 nodes", "examples/grpo_32B_multi_nodes.sh", 5, 5, "#SBATCH --nodes=8"),
]

try:
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=7", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
except Exception:  # noqa: BLE001
    head = "unknown"
print(f"fastrl checkout: {ROOT}  HEAD={head}  (paper cites bce3df7)\n")

fails = 0
for claim, rel, lo, hi, needle in CLAIMS:
    path = os.path.join(ROOT, rel)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        window = "".join(lines[lo - 1:hi])
        ok = needle in window
    except OSError:
        ok = False
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {claim:<46} {rel}:{lo}-{hi}")
print(f"\n{len(CLAIMS) - fails}/{len(CLAIMS)} code claims verified")
sys.exit(1 if fails or head != "bce3df7" else 0)
