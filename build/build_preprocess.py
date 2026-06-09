#!/usr/bin/env python3
"""
build_preprocess.py — FORMAT-ONLY pre-processing for the PDF build.
Changes NO claim, number, quote, tag, or caption *text*. It only:
  (1) lifts the leading H1 title into YAML metadata (title/author/date+build-trace);
  (2) turns each "![short alt](figures/NAME.svg)\n*Figure N.* CAPTION" block into a
      single pandoc implicit-figure "![CAPTION](figures/NAME.pdf){width=90%}" so the
      figure is numbered and listed (List of Figures). The .svg is swapped to the
      vector .pdf (LaTeX cannot embed SVG). Brackets in CAPTION are backslash-escaped
      so provenance tags ([Paper ...], [Dashboard ...], [Code: ...], [Author synthesis])
      and "SCHEMATIC" labels render as LITERAL bracketed text and are never re-parsed
      as links/spans — i.e. no tag can be dropped. The leading "*Figure N.*" is removed
      because pandoc re-adds "Figure N:" (numbering preserved, document order unchanged);
  (3) adds a caption to the single §6.2 results table so it is numbered / in the LoT.
Input: paper.md   Output: paper.build.md
"""
import os, re, datetime

src = open("paper.md", encoding="utf-8").read()
lines = src.split("\n")

title = "Taming the Long-Tail — Asynchronous-Style Speedup (Independent Analysis)"
if lines and lines[0].startswith("# "):
    title = lines[0][2:].strip()
    lines = lines[1:]
body = "\n".join(lines)

# (2) Merge figure image + "*Figure N.*" caption -> one implicit figure; svg->pdf.
fig_re = re.compile(
    r'^!\[[^\]]*\]\(figures/([A-Za-z0-9_]+)\.svg\)[ \t]*\n\*Figure\s+\d+\.\*[ \t]*(.*)$',
    re.M)
def _merge(m):
    name = m.group(1)
    cap = m.group(2).rstrip()
    cap = cap.replace('[', r'\[').replace(']', r'\]')   # literal brackets -> tags survive
    return f'![{cap}](figures/{name}.pdf){{width=90%}}'
body, n_figs = fig_re.subn(_merge, body)

# (3) Caption the §6.2 consolidated-results table (last row starts "| Overheads |").
out, tbl_done = [], False
for ln in body.split("\n"):
    out.append(ln)
    if (not tbl_done) and ln.strip().startswith("| Overheads |"):
        out.append("")
        out.append(": Consolidated results — TLT vs. baselines (per-step token throughput; conditions and provenance in columns). [§6.2]")
        tbl_done = True
body = "\n".join(out)

mtime = datetime.datetime.fromtimestamp(os.path.getmtime("paper.md")).strftime("%Y-%m-%d %H:%M")
today = datetime.date.today().strftime("%Y-%m-%d")
author = "Independent analysis and synthesis (not the original ASPLOS paper) — see provenance note"
yaml = ("---\n"
        f'title: "{title}"\n'
        f'author: "{author}"\n'
        f'date: "Built {today} from paper.md (source mtime {mtime})"\n'
        "---\n\n")

open("paper.build.md", "w", encoding="utf-8").write(yaml + body)
print(f"paper.build.md written | figures merged: {n_figs} | table caption: {tbl_done}")
