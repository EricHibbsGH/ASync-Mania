# ASync-Mania

An independent, fully cited analysis of asynchronous and non-blocking speedup techniques in reasoning reinforcement-learning (RL) training, anchored on the TLT / FastRL system.

Live site: https://erichibbsgh.github.io/ASync-Mania/

## What this is

This repository hosts an independent analysis and synthesis paper. It is not the original ASPLOS paper. It studies how *Taming the Long-Tail* (TLT) accelerates reasoning-RL training by overlapping otherwise-idle work in time rather than by approximating the algorithm: non-blocking drafter training, opportunistic preemptible scheduling, selective asynchronous checkpointing, and tail-phase adaptive speculative decoding. The thesis is that you can capture asynchronous-style GPU-utilization gains while keeping training token-level lossless and strictly on-policy.

The analysis is anchored on the paper and grounded in the released code:

- Source paper: Hu, Yang, Guo, Yao, Lin, Gu, Cai, Gan, Klimovic, Han. *Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter.* ASPLOS '26. arXiv:2511.16665v2. License CC-BY-4.0.
- Released code: `mit-han-lab/fastrl` (Apache-2.0).
- Interactive companion dashboard by Eric Hibbs, whose "Model Math" sections supply the analytical (assumption-based) models reused here.

All original empirical results belong to the paper's authors.

## Provenance: a four-axis tag system

Every quantitative or mechanistic claim in the paper carries one of four tags so a reader always knows where it came from:

- `[Paper: §/Fig/Tbl]`  measured in the source paper.
- `[Dashboard: Model Math §N]`  modeled and assumption-based (Eric Hibbs), not a paper result.
- `[Code: path]`  confirmed in the released fastrl source.
- `[Author synthesis]`  this document's own derivation or framing.

The discipline is strict: a dashboard model is never upgraded into a "paper result," and a code detail is never presented as a measured number.

## Headline of the verification work

The paper was built and checked across seven phases (artifacts in `analysis/`):

- Code grounding: all 11 mechanisms the paper claims were located and confirmed in the released source. The audit surfaced several details the paper rounds or omits, for example a five-state worker machine in code (IDLE / GENERATING / RELEASED / TRAINING / COMPLETED) versus the paper's three-state shorthand; the exact BEG-MAB reward `reward = stable_accept_length * batch_size / total_time` with `a = sum(accept_len)/batch_size + 1`; the elastic speculative-decoding threshold default of 32 plus a 10-check warmup; and the most important reproducibility caveat, that online drafter training ships off by default (`enable_drafter_training=false`).
- Math hardening: every equation was re-derived. One dashboard figure was corrected: the Pareto expected-max-to-median ratio at alpha=2, B=128 is about 14x (exact and Monte-Carlo), not the dashboard's stated 5.7x.
- Integrity: roughly 118 individually sourced claims; 15 verbatim quotes, each under 15 words (longest 8); nothing fabricated. Final verification: PASS (`analysis/FINAL_VERIFICATION.md`).

## Read the paper

- PDF: `paper/ASync-Mania.pdf` (single-column, US Letter, 24 pages).
- Markdown source (frozen content): `paper/paper.md`.
- LaTeX source used to build the PDF: `paper/ASync-Mania.tex`.

## Repository layout

```
ASync-Mania/
  README.md            this file
  LICENSE              CC-BY-4.0 (document), matching the source paper's license
  paper/               ASync-Mania.pdf, paper.md, ASync-Mania.tex
  figures/             11 figures (pdf + svg) and manifest.md
  analysis/            data_table.md and the phase/verification artifacts
  build/               reproducible build (build.ps1 + helpers)
  docs/                GitHub Pages site
```

## Reproduce the PDF

From a machine with pandoc, a LaTeX engine (lualatex via MiKTeX or TeX Live), and Python 3:

```powershell
pwsh build/build.ps1
```

The script reorganizes the frozen `paper.md` into canonical section order, then renders `paper/ASync-Mania.pdf`. Figures are regenerated from inline data arrays by `build/make_figures.py`.

## Links

- Live site: https://erichibbsgh.github.io/ASync-Mania/
- Original paper: https://arxiv.org/abs/2511.16665
- fastrl code (Apache-2.0): https://github.com/mit-han-lab/fastrl
- Interactive dashboard: https://erichibbsgh.github.io/Taming-the-Long-Tail-an-interactive./

## Licensing and attribution

- This document (text, figures, and analysis) is licensed CC-BY-4.0, consistent with the source paper's CC-BY-4.0 license. See `LICENSE`.
- The original experimental results are the work of Hu, Yang, et al. and are theirs.
- Referenced code (`mit-han-lab/fastrl`) is Apache-2.0 and belongs to its authors.
- Analytical models are adapted from the interactive companion dashboard by Eric Hibbs.

If you use this analysis, please cite both the original paper (arXiv:2511.16665) and this repository.
