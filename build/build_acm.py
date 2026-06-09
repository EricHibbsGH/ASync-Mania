#!/usr/bin/env python3
"""
build_acm.py — REORGANIZE (move/merge only; no content edits) paper.md into the
canonical ACM/ASPLOS section order, and emit build sources:
  - paper_canon.md     : full single-column source (YAML title + abstract section + body)
  - async_body.md      : acmart body (no YAML, no abstract section) — abstract is front matter
  - async_abstract.md  : the abstract paragraph alone (-> acmart \begin{abstract})

Reorg performed (verified safe: no in-text cross-ref targets §7.3/§10/§11):
  * §7.3 "Positioning vs async / long-tail RL systems"  ->  new "## 10. Related Work"
  * old "## 10. Extensions and Open Problems" + "## 11. Conclusion"  ->  merged into
    one "## 11. Conclusion" (extensions/open-problems become the future-directions lead).
  * a brief "## Acknowledgements" (framing scaffolding) is added after the Conclusion.
Figures: "![alt](figures/X.svg)\n*Figure N.* CAP"  ->  "![CAP](figures/X.pdf){width=92%}"
  with [ ] escaped so provenance tags render literally and are never dropped.
The §6.2 results table gets a caption. NO claim/number/quote/tag/equation is changed.
"""
import os, re, datetime

src = open("paper.md", encoding="utf-8").read()

# --- split off the H1 title + preamble (subtitle + provenance blockquote) ---
m = re.match(r"# (.*)\n", src)
title = m.group(1).strip()
rest = src[m.end():]

# preamble is everything before "## Abstract"; capture the provenance blockquote (starts "> ")
pre, _, after = rest.partition("## Abstract")
prov_block = "\n".join(l for l in pre.splitlines() if l.startswith(">")).strip()

# --- parse "## " sections from (Abstract onward) ---
chunks = re.split(r"(?m)^## ", "## " + ("Abstract" + after))
sec = {}
for c in chunks:
    if not c.strip():
        continue
    head, _, body = c.partition("\n")
    sec[head.strip()] = body

def figconv(text):
    fig_re = re.compile(r"^!\[[^\]]*\]\(figures/([A-Za-z0-9_]+)\.svg\)[ \t]*\n\*Figure\s+\d+\.\*[ \t]*(.*)$", re.M)
    def repl(mm):
        cap = mm.group(2).rstrip().replace("[", r"\[").replace("]", r"\]")
        return f"![{cap}](figures/{mm.group(1)}.pdf){{width=92%}}"
    return fig_re.sub(repl, text)

def tablecap(text):
    out, done = [], False
    for ln in text.split("\n"):
        out.append(ln)
        if (not done) and ln.strip().startswith("| Overheads |"):
            out.append("")
            out.append(": Consolidated results: TLT vs. baselines (per-step token throughput; conditions and provenance in columns). [§6.2]")
            done = True
    return "\n".join(out)

# --- split §7 into (7.1+7.2 keep) and (7.3 -> Related Work) ---
s7 = sec["7. Why Not Fully Async?"]
i73 = s7.find("### 7.3")
s7_keep = s7[:i73].rstrip()
s73 = s7[i73:]
# drop the "### 7.3 ..." heading line, keep its body for Related Work
related_body = s73.split("\n", 1)[1].strip()

# --- merged Conclusion = Extensions/Open-Problems (future directions) + Conclusion ---
ext = sec["10. Extensions and Open Problems"].strip()
concl = sec["11. Conclusion"].strip()
merged_concl = (
    "Future directions: extensions and open problems. "
    "The companion dashboard proposes three forward-looking variants, design sketches "
    "*not* evaluated by the paper, inheriting the §5 assumptions.\n\n"
    + ext + "\n\nConcluding remarks. " + concl
)

acks = ("This is an independent analysis. We gratefully acknowledge Hu, Yang, Guo, Yao, Lin, Gu, "
        "Cai, Gan, Klimovic, and Han for the TLT/FastRL work (ASPLOS '26, arXiv:2511.16665v2, CC-BY-4.0) "
        "and the released `mit-han-lab/fastrl` code (Apache-2.0); all original empirical results are theirs. "
        "Analytical models are adapted from the interactive companion dashboard by Eric Hibbs.")

# --- assemble body in canonical order (manual section numbers kept) ---
order = [
    ("## 1. Introduction", sec["1. Introduction"]),
    ("## 2. Background and Motivation", sec["2. Background"]),
    ("## 3. The Long-Tail, Formally", sec["3. The Long-Tail, Formally"]),
    ("## 4. Mechanisms of Async Speedup", sec["4. Mechanisms of Async Speedup"]),
    ("## 5. A Mathematical Model of the Speedup", sec["5. A Mathematical Model of the Speedup"]),
    ("## 6. Evaluation Synthesis", sec["6. Evaluation Synthesis"]),
    ("## 7. Why Not Fully Async?", "\n" + s7_keep + "\n"),
    ("## 8. Implementation & Reproducibility", sec["8. Implementation & Reproducibility"]),
    ("## 9. Threats to Validity / Assumptions Ledger", sec["9. Threats to Validity / Assumptions Ledger"]),
    ("## 10. Related Work", "\n" + related_body + "\n"),
    ("## 11. Conclusion", "\n" + merged_concl + "\n"),
    ("## Acknowledgements {-}", "\n" + acks + "\n"),
    ("## Figures {-}", figconv(sec["Figures"])),
    ("## References {-}", sec["References"]),
]
body = prov_block + "\n\n" + "\n".join(h + "\n" + tablecap(b) for h, b in order)

abstract_md = sec["Abstract"].strip()

# --- write single-column source (with YAML) ---
mtime = datetime.datetime.fromtimestamp(os.path.getmtime("paper.md")).strftime("%Y-%m-%d %H:%M")
today = datetime.date.today().strftime("%Y-%m-%d")
author = "Eric Hibbs, independent analysis (not the original ASPLOS paper); see provenance note"
yaml = ("---\n" f'title: "{title}"\n' f'author: "{author}"\n'
        f'date: "Built {today} from paper.md (frozen content; mtime {mtime})"\n' "---\n\n")
open("paper_canon.md", "w", encoding="utf-8").write(yaml + "## Abstract\n\n" + abstract_md + "\n\n" + body)

# --- write acmart pieces ---
open("async_body.md", "w", encoding="utf-8").write(body)
open("async_abstract.md", "w", encoding="utf-8").write(abstract_md + "\n")

print("build_acm.py OK | title:", title[:50])
print("sections:", [h for h,_ in order])
