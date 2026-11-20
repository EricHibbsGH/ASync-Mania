# Verification package

This folder holds scripts that let anyone re-check the paper's numbers, its code citations, and its one executed test without trusting the author's write-up.

| Script | What it checks | Needs | Result at submission |
|---|---|---|---|
| `verify_numbers.py` | Recomputes all 44 derived quantities (ratios, geometric means, Pareto order statistic incl. a seeded Monte Carlo run, SD model optima, Amdahl bound, cost arithmetic) and compares each with the value printed in the paper | Python 3, numpy | 44/44 PASS (`verify_numbers_results.txt`) |
| `verify_code_claims.py` | Checks all 41 file:line code citations against a `fastrl` checkout at commit `bce3df7` | Python 3, git, a fastrl clone | 41/41 PASS (`verify_code_claims_results.txt`) |
| `test_drafter_masking.py` | Runs fastrl's own Qwen2 drafter class the way the online trainer calls it, and tests causality and cross-sample attention. A positive control runs the same harness on the stock `transformers` model | Python 3, torch (CPU is fine), `transformers==4.57.1`, a fastrl clone | Deployed path (eager attention): not causal, cross-sample attention in both directions. Positive control: causal (`test_drafter_masking_results.txt`) |

## Run

```bash
git clone https://github.com/mit-han-lab/fastrl && git -C fastrl checkout bce3df7
python verify_numbers.py
python verify_code_claims.py fastrl
python -m venv venv && venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu "transformers==4.57.1"
venv/bin/python test_drafter_masking.py fastrl
```

On Windows, use `venv\Scripts\...`, and create the venv on a short path: torch's install exceeds the 260-character path limit if the path is deep.

## Limits

- The numeric and code checks confirm internal consistency against the cited sources. They are not an independent experimental replication of TLT.
- The masking test uses a tiny random-weight model. It establishes the attention behavior of the code path, not the effect on a trained drafter's accept length. Other `transformers` versions may take a different mask branch; `transformers==4.57.1` is the version pinned by fastrl's vendored SGLang runtime.
