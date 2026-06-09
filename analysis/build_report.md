# Build Report — `Taming-the-Long-Tail_Async-Analysis.pdf`

**Build pass only.** Content was frozen; this report covers rendering and the Phase-C integrity gate. No claim, number, quote, citation, figure, or provenance tag was changed.

## Toolchain (detected & used)
| Tool | Version | Role |
|---|---|---|
| pandoc | 3.10 (`C:\Users\bojo1\AppData\Local\Pandoc\pandoc.exe`) | Markdown → LaTeX → PDF |
| LaTeX engine | **lualatex**, MiKTeX 25.12 | typesetting (chosen primary path) |
| pdftotext | 24.04.0 (MiKTeX) | Phase-C text extraction / verification |
| Python | 3.12 | format-only preprocessing |

Primary path (pandoc + lualatex) succeeded; the WeasyPrint fallback was **not** needed.

## Exact build command
Reproducible via `build.ps1` (PowerShell). Core invocation:
```powershell
python build_preprocess.py            # format-only: H1->title, table caption
pandoc paper.build.md `
  --pdf-engine=lualatex --lua-filter="mathfix.lua" `
  --toc --toc-depth=3 -V lof=true -V lot=true `
  -V documentclass=article -V papersize=letter -V fontsize=11pt -V geometry:margin=1in `
  -V colorlinks=true -V linkcolor=Blue -V urlcolor=Blue -V toccolor=Black -V linestretch=1.05 `
  --include-in-header="header.tex" --include-before-body="frontmatter.tex" `
  -o "Taming-the-Long-Tail_Async-Analysis.pdf"
```
pandoc runs lualatex enough passes to resolve the TOC / LoF / LoT and cross-references.

## Format-only transforms (no content change)
- **`build_preprocess.py`** — lifts the H1 into a YAML title block (so there is a title/header page) and adds one caption to the §6.2 results table (so it is numbered and appears in the List of Tables). It also merges each `![alt](…svg)` + `*Figure N.*` caption into one pandoc implicit figure pointing at the **vector `.pdf`**, with `[`/`]` backslash-escaped so the bracketed provenance tags render as **literal** text and can never be re-parsed/dropped.
- **`mathfix.lua`** — inside math spans only, rewrites the markdown-escaped `\*` back to `*` so the `k^*` / `\sigma^*` ("star") equations typeset. Math-only; equation meaning unchanged.
- **`header.tex`** — Latin Modern body via fontspec; a `\newunicodechar` net maps all 16 math-ish non-ASCII glyphs (×, →, ⇒, ≈, ≤, −, ∈, ∎, ·, …, ⁵, α, γ, τ, ε, Γ) to real LaTeX glyphs; §/—/–/ï covered natively. `xurl` breaks long URLs.
- **`frontmatter.tex`** — the prominent "independent analysis, not the original ASPLOS paper" statement + the four-axis provenance legend box (metadata framing).

## Phase-C integrity gate
Source `paper.md` vs extracted PDF text (`pdf_text.txt`, default pdftotext):

| Check | Result | Verdict |
|---|---|---|
| Provenance tags `[Paper` | md 110 → pdf 128 | ✅ no drop (surplus = List of Figures repeats figure-caption tags + legend) |
| Provenance tags `[Dashboard` | md 23 → pdf 28 | ✅ no drop (same reason) |
| Provenance tags `[Code` | md 22 → pdf 23 | ✅ no drop (+1 legend) |
| Provenance tags `[Author` | md 10 → pdf 11 | ✅ no drop (+1 legend) |
| Figures embedded (vector PDF) | 11 "Figure N:" captions | ✅ = manifest (11) |
| List of Figures / List of Tables / Table 1 | all present | ✅ |
| Verbatim quotes (Phase-3 set of 15) | 14/15 via pdftotext; **15/15 present in PDF** | ✅ (see note 2) |
| Headline numbers | 1.86×, 2.12×, ~85%, 893/280/97 ms, 7.81/30.39/10.69 GB, 4.59→6.53, 3.76→5.15, 3.66×, 9.78, 2.44× | ✅ all present |
| Corrected Pareto ratio | **14×** present (and 5.7× also present, shown as the flagged-wrong dashboard value) | ✅ reads 14×, not 5.7× |
| Section parity (14 top-level headings) | 0 missing from PDF | ✅ |
| Pages | 24 | — |

**Note 1 — tag counts are *higher* in the PDF, not equal.** Every tag in `paper.md` is present (pdf ≥ md on all four axes); the surplus comes entirely from (a) the **List of Figures**, which verbatim-repeats the 11 figure captions (several carry `[Paper]`/`[Dashboard]` tags), and (b) the **provenance legend box**, which names all four tags once. **No tag was dropped** — the success criterion ("render dropped nothing") is met; exact equality is impossible once the LoF legitimately duplicates captions.

**Note 2 — "training-time test" and pdftotext.** This quote did not match under default pdftotext because its hyphen falls at a line break and pdftotext's de-hyphenation strips it on *extraction* ("trainingtime"). It is **rendered correctly in the PDF**: pandoc's LaTeX output contains `multi-pass ``training-time test`` verbatim, and LaTeX never drops explicit hyphens. This is an extraction-tool artifact, not a content loss. All 15 quotes are faithful in the document.

## Issues encountered & resolved (build side only)
1. `luaotfload` font-fallback (`add_fallback`/`RawFeature`) errored on this MiKTeX build ("index a nil value") → removed; relied on the explicit `\newunicodechar` net (covers every non-ASCII glyph in `paper.md`).
2. `DejaVu Sans Mono` not registered for fontspec → switched mono to **Latin Modern Mono**.
3. LaTeX math error on `k^\*` (markdown-escaped asterisk passed verbatim into `$…$`) → `mathfix.lua` restores `*` in math only.
4. MiKTeX prints a harmless "you have not checked for MiKTeX updates" nag to stderr → `build.ps1` uses `ErrorActionPreference=Continue` and judges success by the PDF existing.

## Verdict
**PDF is a faithful, complete render of the verified paper.md.** 24 pages, 11 embedded vector figures, all math typeset, TOC + List of Figures + List of Tables, four-axis provenance intact, references with corrected "31st" BibTeX + DOI + dashboard/repo artifacts.
