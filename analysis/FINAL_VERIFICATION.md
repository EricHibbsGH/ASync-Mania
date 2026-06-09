FINAL VERIFICATION: PASS

# Final Verification Gate (Phase A)

Re-confirmation (not re-derivation) of the frozen content against the current `paper.md`
and the built PDF `Taming-the-Long-Tail_Async-Analysis.pdf` (24 pages). Extraction:
`pdftotext -enc UTF-8`. Date: 2026-06-07.

| Check | Expected | Result | Verdict |
|---|---|---|---|
| Tag `[Paper` | pdf ≥ md | md 110 → pdf 128 | ✅ PASS |
| Tag `[Dashboard` | pdf ≥ md | md 23 → pdf 28 | ✅ PASS |
| Tag `[Code` | pdf ≥ md | md 22 → pdf 23 | ✅ PASS |
| Tag `[Author` | pdf ≥ md | md 10 → pdf 11 | ✅ PASS |
| Figures (`Figure N:` captions) | 11 | 11 | ✅ PASS |
| List of Figures / List of Tables | present | both present | ✅ PASS |
| Verbatim quotes (15) | 15 faithful | 14/15 via pdftotext; 15/15 in PDF | ✅ PASS (see note) |
| Headline numbers | all present | 1.86×, 2.12×, ~85%, 893/280/97 ms, 7.81/30.39/10.69 GB, 4.59→6.53, 3.76→5.15, 3.66×, 9.78, 2.44×, 14× | ✅ PASS |
| Corrected Pareto ratio | reads 14× (not 5.7×) | 14× present; 5.7× present only as the flagged-wrong dashboard value | ✅ PASS |
| Section parity | all top-level headings | 14/14 present, 0 missing | ✅ PASS |

**Tag-count note.** PDF counts are *higher* than `paper.md`, never lower — so no tag was
dropped. The surplus is entirely from (a) the **List of Figures**, which verbatim-repeats
the 11 figure captions (several carrying `[Paper]`/`[Dashboard]` tags), and (b) the
**provenance legend box** (names each of the four tags once). pdf ≥ md on all four axes is
the correct "nothing dropped" criterion.

**Quote note.** "training-time test" is the one quote not recovered by default `pdftotext`:
its hyphen falls at a line break and pdftotext's de-hyphenation strips it on *extraction*
("trainingtime"). It is rendered correctly in the PDF — pandoc's LaTeX output contains
``multi-pass ``training-time test`` verbatim and LaTeX never drops explicit hyphens. This
is an extraction-tool artifact, **not** a content loss; it must not be re-flagged.

**No mismatches found. Phase A passes; proceeding to Phase B (ACM reorganization).**
