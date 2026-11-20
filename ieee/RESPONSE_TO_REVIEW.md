# Response to external review (2026-11-20)

This document answers the external review point by point. For each point it gives the change made and where the evidence is. Section numbers refer to the revised `main.pdf`.

## Findings

| Review point | Response | Where |
|---|---|---|
| §2, §8: possible AI-assisted writing | **Confirmed and now disclosed.** Large language model tools (including Anthropic's Claude) were used to draft and edit text, re-check transcribed values, help trace code references, and write the figure and verification scripts. A disclosure following IEEE policy on AI-generated content is in the Acknowledgment. Claims no longer rest on the prose: every number and code citation is checked by scripts anyone can run. | Acknowledgment; `verification/` |
| §3: KL/Fisher relation should be marked as heuristic | Done. The text now states that Inequality (6) is "a heuristic tracking model built on the two stated approximations, not a convergence theorem". | Sec. VI-D |
| §3: Pareto, SD model and Amdahl depend on assumptions | These were already stated, and are now also enumerated in Threats to Validity. All three are re-computed by `verify_numbers.py`. | Secs. IV, VI, XI |
| §4: code differences depend on revision and interpretation | All 41 citations are pinned to `bce3df7` and machine-checked by `verify_code_claims.py` (41/41 PASS). | Sec. IX; `verify_code_claims_results.txt` |
| §5: sequence-packing issue was static only and should be reproduced | **Now executed.** `test_drafter_masking.py` runs fastrl's drafter class exactly as the trainer calls it, under the pinned `transformers==4.57.1`. Result: the deployed path (eager attention plus fastrl's Boolean fallback mask) is **not causal**, and packed samples attend to each other in both directions, even with positions reset. A positive control (the stock HF model) is exactly causal, which validates the harness. The earlier static description ("a plain causal mask") was wrong and has been corrected. | New Sec. IX-A |
| §6: not an experimental replication | Now stated explicitly: "No training runs or performance measurements were made for this paper; it is not an experimental replication of TLT." | Sec. XI, Audit limits |
| §7: 3.64× vs. 3.66× explained as "separate runs" without evidence | The speculation was removed: "the source does not say whether these come from separate runs … the 0.5% difference is left unexplained." | Sec. XI |
| §7: underlying measurements should be documented | Precision notes added: the A100 geometric mean is 1.79× as printed in the source, but 1.78× when recomputed from the rounded per-model labels. The Monte Carlo value is now seeded and reproducible (20.12). | Secs. IV, VII-B, Fig. 5 |
| §9: assumptions worth defending | Each assumption is listed in Threats to Validity with the evidence that contradicts it or is missing. The models are used only qualitatively. | Sec. XI |
| §10: author verification is not independent verification | Agreed. The scripts exist so that a reviewer can verify independently, without trusting the author's documents. | `verification/README.md` |

## Recommended questions

| # | Question | Answer |
|---|---|---|
| 1 | Drafts and revision history | The public git history of `EricHibbsGH/ASync-Mania` records the paper's versions, and the analysis logs are in `analysis/`. *(Author: add any earlier drafts or notes you hold.)* |
| 2 | Were AI tools used, and for what? | Yes. See the Acknowledgment disclosure and §2 above. |
| 3 | Pareto calculation and the 5.7× error | Sec. IV; `verify_numbers.py` lines "Pareto …". 5.7 = √128/2 = B^(1/α)/α, which drops the tail constant and mis-normalizes by the median. |
| 4 | Amdahl calculation and its assumptions | Eq. (7) inverted at S = 1.66 and 2.12 with f = 0.85 gives s = 1.88 and 2.64 (`verify_numbers.py`). Assumption: f transfers across configurations (Sec. XI). |
| 5 | Demonstrate the five implementation differences | `verify_code_claims.py`, 41/41 at `bce3df7`. |
| 6 | Reproduce the sequence-packing issue with a targeted test | Done: `test_drafter_masking.py` (Sec. IX-A). |
| 7 | Why online drafter training is off by default, and how to enable it | The reason is not stated by the TLT authors. Enabling it requires `enable_drafter_training: true` and `collect_hidden_states_from_sgl: true` (Sec. X). Note that the masking defect in Sec. IX-A should be fixed before relying on that path. |
| 8 | Which contributions are original | New paragraph "What is original here" in Sec. III. |
| 9 | An independent reproduction of a key result | The masking behavior is reproduced by execution. All derived numbers are reproducible by script. TLT's performance numbers were not re-measured (no GPU cluster). |

## Not changed, and why

- **The paper's prose was not rewritten to evade AI detectors.** Detector scores are unreliable either way. The appropriate response to AI-assisted writing is disclosure plus verifiable evidence, which this revision provides.
- **Responsible disclosure.** The masking defect is in a public open-source project. Before or at publication, open an issue on `mit-han-lab/fastrl` that links `test_drafter_masking.py`, so the maintainers can respond.
