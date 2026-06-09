# Phase 5 — Citation & Source Integrity

Confirms every reference cited in `paper.md` resolves, that none is invented, and that no citation is used to support a claim it doesn't make. Resolve sources: 📄 = TLT paper's own bibliography (read verbatim from the PDF, pp. 14–16); 🌐 = independently web-verified (Phase 2); 💻 = on-disk repo (Phase 1).

## A. Primary / anchor artifacts (full, corrected metadata)

| Artifact | Canonical details | Status |
|---|---|---|
| **TLT paper (anchor)** | Hu, Yang, Guo, Yao, Lin, Gu, Cai, Gan, Klimovic, Han. *Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter.* **ASPLOS '26**, Pittsburgh, PA, **Mar 22–26 2026**. arXiv:**2511.16665v2** (21 Jan 2026). **DOI 10.1145/3779212.3790231**. ACM ISBN 979-8-4007-2359-9. | ✓ 📄/🌐 |
| **Code** | `github.com/mit-han-lab/fastrl`, **Apache-2.0**, `main` HEAD `bce3df7`; HF: `mit-han-lab/Qwen2.5-{7B,32B}-Eagle-RL` | ✓ 💻 |
| **Dashboard (secondary)** | Eric Hibbs, *Taming the Long-Tail — interactive companion*, "Model Math" §1–6. `https://erichibbsgh.github.io/Taming-the-Long-Tail-an-interactive./` (accessed 2026-06-05) | ✓ 🌐 |

**BibTeX corrections applied (Phase 2 §E):**
- README `@inproceedings{TLT}` has **`31th`** (typo) → use **`31st`** + full proceedings string + DOI.
- README **News-line** arXiv link text uses a *non-canonical* title; the **PDF title is canonical** and is used throughout.

## B. External works cited in `paper.md` (resolve status)

| Ref | Work | ID / venue | Status | Claim it supports — fair? |
|---|---|---|---|---|
| GRPO | DeepSeekMath (Shao et al.) | arXiv:2402.03300 | ✓ 📄 | group-relative baseline — FAIR |
| DeepSeek-R1 | DeepSeek-AI | arXiv:2501.12948 | ✓ 📄 | reasoning RL pipeline — FAIR |
| PPO | Schulman et al. | CoRR 2017 | ✓ 📄 | critic-based — FAIR |
| Leviathan SD | Fast Inference via Speculative Decoding | ICML'23 | ✓ 📄 | lossless SD — FAIR |
| Chen et al. SD | Accelerating LLM Decoding w/ Speculative Sampling | arXiv:2302.01318 | ✓ 📄 | lossless SD — FAIR |
| EAGLE | Li et al. | ICML'24 | ✓ 📄 | feature-level autoregressive drafter — FAIR |
| EAGLE-2 | Li et al. | EMNLP'24 | ✓ 📄 | dynamic draft trees — FAIR |
| EAGLE-3 | Li et al. | arXiv:2503.01840 | ✓ 📄 | multi-layer fusion + training-time test — FAIR |
| HASS | Zhang et al. | ICLR'25 | ✓ 📄 | multi-step training-time test — FAIR |
| Medusa | Cai et al. | arXiv:2401.10774 | ✓ 📄 | multiple heads — FAIR |
| Hydra | Ankner et al. | COLM'24 | ✓ 📄 | sequentially-dependent heads — FAIR |
| SpecInfer | Miao et al. | ASPLOS'24 | ✓ 📄 | tree-based verification — FAIR |
| OSD | Liu et al. | ICML'24 | ✓ 📄 | KD/reverse-KD online drafter — FAIR |
| Lookahead | Zhao et al. | KDD'24 | ✓ 📄 | model-free retrieval drafting [72] — FAIR |
| REST | He et al. | arXiv:2311.08252 | ✓ 📄 | retrieval-based SD — FAIR |
| HybridFlow/VeRL | Sheng et al. | EuroSys'25 | ✓ 📄 | SOTA baseline; orchestration focus — FAIR |
| Open-R1 | Hugging Face | 2025 | ✓ 📄 | baseline (TRL+vLLM+DeepSpeed) — FAIR |
| RLHFuse | Zhong et al. | NSDI'25 | ✓ 📄 | stage fusion — FAIR |
| ReaLHF | Mei et al. | arXiv:2406.14088 | ✓ 📄 | parameter reallocation — FAIR |
| **AReaL** | Fu et al. | arXiv:2505.24298 | ✓ 📄/🌐 | asynchronous RL — FAIR |
| **StreamRL** | Zhong et al. | arXiv:2504.15930 | ✓ 📄/🌐 | disaggregated streaming gen — FAIR |
| **APRIL** | Zhou et al. | arXiv:2509.18521 | ✓ 🌐 | active partial rollouts — FAIR |
| **RollPacker** | Gao et al. | arXiv:2509.21009 | ✓ 🌐 | tail batching (**synchronous**) — FAIR (see Phase 2 NEEDS-EDIT: don't call it async) |
| SGLang | Zheng et al. | NeurIPS'24 | ✓ 📄/💻 | rollout backend — FAIR |
| Ray | Moritz et al. | OSDI'18 | ✓ 📄/💻 | distributed execution — FAIR |
| ZeroMQ | Hintjens | O'Reilly 2013 | ✓ 📄/💻 | coordinator transport — FAIR |
| FSDP2 / DCP | PyTorch Team | 2025 | ✓ 📄/💻 | training / async ckpt — FAIR |
| Sequence packing | Krell et al. | arXiv:2107.02027 | ✓ 📄 | varlen packing — FAIR |
| Qwen2.5 | Qwen et al. | arXiv:2412.15115 | ✓ 📄 | target models — FAIR |
| Eurus-2-RL | Cui et al. | arXiv:2502.01456 | ✓ 📄 | RL dataset — FAIR |
| OpenThoughts2-1M | OpenThoughts | 2025 | ✓ 📄/💻 | drafter warm-up — FAIR |
| MT-Bench | Zheng et al. | CoRR 2023 | ✓ 📄 | downstream eval — FAIR |
| DAPO | Yu et al. | arXiv:2503.14476 | ✓ 📄 | compatible RL algo — FAIR |
| RLOO | Ahmadian et al. | ACL'24 | ✓ 📄 | compatible RL algo — FAIR |
| REINFORCE++ | Hu et al. | arXiv:2501.03262 | ✓ 📄 | compatible RL algo — FAIR |
| Adam | Kingma & Ba | ICLR'15 | ✓ 📄 | optimizer — FAIR |
| ByteDance trace | W&B DAPO reproduction | 2025 | ✓ 📄 | Fig 2 trace source — FAIR |

## C. New citation added in Phase 6 (search-sourced, dated)

| Work | ID | Status | Use |
|---|---|---|---|
| MIT News, *"New method could increase LLM training efficiency"* | news.mit.edu/2026/…-0226 (2026-02-26) | ✓ 🌐 | "Reception & follow-on work" note only |
| *Beat the long tail: Distribution-Aware Speculative Decoding for RL Training* | arXiv:2511.13841 | ✓ 🌐 | concurrent-work mention only (not a TLT result) |

## D. Integrity findings
- **No invented references:** every entry above resolves to the TLT bibliography, the on-disk repo, or a web-verified source.
- **No citation overreach:** each external citation supports only the narrow claim attributed to it (verified in Phase 2). The "neglect rollout" framing is attributed to the **paper's** positioning, not asserted as fact.
- **One correction propagated to references:** "31st" (not "31th"); canonical PDF title; DOI included.
- The dashboard and the repo are cited as first-class artifacts with access date / commit.
