# Phase 0 — Inventory & Environment

Date of pass: 2026-06-05. Working dir: `c:\Users\bojo1\OneDrive\Desktop\ASNYC - Eric Hibbs`.

## Deliverables from the prior step (present, read)

| File | Size | Status |
|---|---|---|
| `paper.md` | 40.9 KB | Present, read |
| `data_table.md` | 15.9 KB | Present, read |
| `provenance_check.md` | 9.2 KB | Present, read |

## Primary sources

| Source | Reachable? | Notes |
|---|---|---|
| **Paper PDF** | **Not on disk** | The paper was supplied as a chat attachment, not a file in the working dir (`Glob *.pdf` → none). Its full text (ASPLOS '26, arXiv:2511.16665**v2**, 21 Jan 2026) is available verbatim in context and is treated as the canonical source. Implication: every `[Paper …]` tag traces to the attached v2 text, not a local file. |
| **Dashboard** | ✅ Reachable | `https://erichibbsgh.github.io/Taming-the-Long-Tail-an-interactive./` — fetched twice in the prior step; Model Math §1–6 equations verified consistent across both fetches. |
| **Code repo** | ✅ Cloned | See below. |

## Code acquisition

- `git ls-remote` → reachable; `main` HEAD `bce3df7a4d46473912e9b81bf47bca419729557f`.
- `git clone --depth 1 https://github.com/mit-han-lab/fastrl.git fastrl` → **exit 0**, 2,639 files. (Shallow clone; full history not fetched — not needed for code grounding.)
- **Submodules:** repo has **no `.gitmodules`**. `third-party/sglang` is **vendored in-tree** (not a git submodule), so the SGLang code is present without a recursive fetch. `git clone --recursive` was therefore unnecessary.

### Verified repo facts (checked, not trusted)

| Claimed fact | Verdict |
|---|---|
| Apache-2.0 license | ✅ `LICENSE` = "Apache License Version 2.0" |
| Top-level dirs `verl/ eagle-train/ examples/ scripts/ third-party/ data/ assets/` | ✅ all present (also `LICENSE`, `README.md`, `requirements.txt`, `setup.py`, `.pre-commit-config.yaml`) |
| Entry scripts `examples/bench_sd.sh`, `grpo_7B.sh`, `grpo_32B_multi_nodes.sh` | ✅ all three present (examples/ contains exactly these 3) |
| `third-party/sglang/` present | ✅ full SGLang tree (python/, sgl-kernel/, sgl-router/, benchmark/, test/, …) |
| HF models `mit-han-lab/Qwen2.5-7B/32B-Eagle-RL` | ⏳ to confirm in Phase 1/5 (README links a `mit-han-lab/tlt` HF collection) |
| SGLang PR #4732 (paper ref [59]) | ⏳ to confirm in Phase 1/2 against vendored sglang + the PR online |

### Python-file inventory (grounding surface)

| Area | `.py` files |
|---|---|
| `verl/` | 281 |
| `eagle-train/` | 12 (incl. `eagle_trainer.py`, `eagle3_trainer.py`, `eagle_datagen.py`, `utils.py`) |
| `third-party/sglang/` | 1,287 |

Tooling present: git 2.47.0, Python 3.12.10 (+3.12.2), matplotlib 3.8.3, numpy 1.26.4 → Phase 4 figures are runnable locally.

## Blocked / unavailable

- **Nothing blocked.** Network reachable; full code (incl. SGLang) on disk; dashboard reachable; paper text in context. The only "absence" is the PDF as a *file* (the content is available). No claim need be downgraded for unreachable code in this pass.

## Discrepancies already surfaced (for later phases)

1. **Title mismatch (Phase 2a):** README News line links arXiv with title *"Taming the Long Tail: Efficient Reinforcement Learning for Language Models via Adaptive Speculative Decoding"*; PDF canonical title is *"Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter."* Both recorded; PDF canonical.
2. **MIT coverage (Phase 2):** README confirms "[2026/02] … MIT homepage … cover of MIT News" with URLs `web.mit.edu/archive/spotlight/more-efficient-llms/` and `news.mit.edu/2026/new-method-could-increase-llm-training-efficiency-0226` → to verify/date in Phase 2.

## Verdict

Phase 0 complete. Environment fully enables four-way grounding ([Paper]/[Dashboard]/[Code]/[Author synthesis]). Proceeding to Phase 1 (code grounding) — the core of this pass — using the on-disk repo at `…/fastrl`.
