#!/usr/bin/env python3
"""
acm_postprocess.py — assemble ASync-Mania.tex (acmart sigconf, two-column) from the
pandoc fragments async_abstract.tex + async_body.tex. FORMAT ONLY:
  * wrap in acmart frontmatter (title/author/abstract/CCS/keywords/maketitle);
  * promote every figure -> figure* (full page width); a \clearpage after every 2nd
    figure keeps the 11-figure plate from overflowing the float stack;
  * convert the one pandoc longtable (§6.2 results) -> table* + tabular (longtable cannot
    span columns in two-column); cell text copied verbatim (no number/tag changed);
  * define the pandoc \real helper + a \newunicodechar glyph net (same as header.tex).
No claim/number/quote/tag/equation is altered. No bold, no em-dashes in added scaffolding.
"""
import re

abstract = open("async_abstract.tex", encoding="utf-8").read().strip()
body = open("async_body.tex", encoding="utf-8").read()

# --- convert the single pandoc longtable to table* + tabular ---
def convert_longtable(text):
    lt_m = re.search(r"\\begin\{longtable\}.*?\\end\{longtable\}", text, re.S)
    if not lt_m:
        return text, 0
    lt = lt_m.group(0)
    spec = re.search(r"\\begin\{longtable\}\[\]\{(?P<s>.*)\}\s*\\caption", lt, re.S).group("s")
    cap = re.search(r"\\caption\{(?P<c>.*)\}\s*\\tabularnewline", lt, re.S).group("c")
    header = re.search(r"\\toprule\\noalign\{\}\s*(?P<h>.*?)\\midrule", lt, re.S).group("h").strip()
    rows = re.search(r"\\endlastfoot\s*(?P<b>.*?)\\end\{longtable\}", lt, re.S).group("b").strip()
    tbl = ("\\begin{table*}[t]\n\\centering\\small\n"
           f"\\caption{{{cap}}}\n\\begin{{tabular}}{{{spec}}}\n"
           f"\\toprule\n{header}\n\\midrule\n{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n")
    return text.replace(lt, tbl), 1

body, n_tbl = convert_longtable(body)

# --- figures -> figure*; clearpage after every 2nd to bound the float stack ---
body = body.replace("\\begin{figure}", "\\begin{figure*}[t]").replace("\\end{figure}", "\\end{figure*}")
segs = body.split("\\end{figure*}")
out, cnt = segs[0], 0
for seg in segs[1:]:
    cnt += 1
    out += "\\end{figure*}" + ("\n\\clearpage" if cnt % 2 == 0 else "") + seg
body, n_fig = out, cnt

GLYPHS = (r"\usepackage{newunicodechar}" "\n"
    r"\newunicodechar{×}{\ensuremath{\times}}\newunicodechar{→}{\ensuremath{\rightarrow}}" "\n"
    r"\newunicodechar{⇒}{\ensuremath{\Rightarrow}}\newunicodechar{≈}{\ensuremath{\approx}}" "\n"
    r"\newunicodechar{≤}{\ensuremath{\leq}}\newunicodechar{−}{\ensuremath{-}}" "\n"
    r"\newunicodechar{∈}{\ensuremath{\in}}\newunicodechar{∎}{\ensuremath{\blacksquare}}" "\n"
    r"\newunicodechar{·}{\ensuremath{\cdot}}\newunicodechar{…}{\ldots}" "\n"
    r"\newunicodechar{⁵}{\textsuperscript{5}}\newunicodechar{α}{\ensuremath{\alpha}}" "\n"
    r"\newunicodechar{γ}{\ensuremath{\gamma}}\newunicodechar{τ}{\ensuremath{\tau}}" "\n"
    r"\newunicodechar{ε}{\ensuremath{\varepsilon}}\newunicodechar{Γ}{\ensuremath{\Gamma}}")

PRE = (
r"""\documentclass[sigconf,nonacm]{acmart}
\settopmatter{printacmref=false,printccs=true,printfolios=true}
\setcopyright{none}
\renewcommand\footnotetextcopyrightpermission[1]{}
\setcounter{secnumdepth}{0}
\usepackage{calc}
\providecommand{\real}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\extrafloats{100}
""" + GLYPHS + r"""
\begin{document}
\title{Overlapping, Not Approximating: Asynchronous-Style Speedup in Reasoning-RL Training and the Lossless Design of TLT}
\author{Eric Hibbs}
\affiliation{\institution{Independent Analysis}\city{Online}\country{USA}}
\renewcommand{\shortauthors}{Hibbs (independent analysis)}
\begin{abstract}
""" + abstract + r"""
\end{abstract}
\begin{CCSXML}
<ccs2012>
<concept><concept_id>10010583.10010600</concept_id>
<concept_desc>Computer systems organization~Neural networks</concept_desc>
<concept_significance>500</concept_significance></concept>
</ccs2012>
\end{CCSXML}
\ccsdesc[500]{Computer systems organization~Neural networks}
\keywords{Large Language Models, Reinforcement Learning, Speculative Decoding, Training Efficiency}
\maketitle
\begin{center}\fcolorbox{black}{gray!12}{\parbox{0.95\columnwidth}{\footnotesize
\textit{Independent analysis, not the original ASPLOS paper.} Built on Hu, Yang
\emph{et al.}, \emph{Taming the Long-Tail} (ASPLOS~'26, arXiv:2511.16665v2, CC-BY-4.0),
with analytical models adapted from the interactive companion dashboard by Eric Hibbs and
mechanisms grounded in the released \texttt{mit-han-lab/fastrl} (Apache-2.0).
All original empirical results belong to the paper's authors.}}\end{center}

""")

open("ASync-Mania.tex", "w", encoding="utf-8").write(PRE + body + "\n\\end{document}\n")
print(f"ASync-Mania.tex written | longtable->table*: {n_tbl} | figures->figure*: {n_fig}")
