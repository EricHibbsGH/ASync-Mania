#!/usr/bin/env python3
"""
deslop.py — remove two "AI-slop" style signals from a Markdown file, in place:
  (1) all bold emphasis markers  **like this**  ->  like this
  (2) em-dash / double-hyphen comma-splice breaks  ( em-dash , ---, ' -- ' )  ->  comma

Protects (never alters): fenced code blocks, inline code, $math$/$$math$$,
http(s) URLs, en-dash numeric ranges (1.7-2.1), and ordinary hyphens (memory-bound).
Changes NO numbers, quotes, provenance tags, or equations, only emphasis markers and
em-dash punctuation. Usage: python deslop.py <file.md>
"""
import sys, re

path = sys.argv[1]
text = open(path, encoding="utf-8").read()

# --- carve out protected spans (code fences, inline code, math, urls) ---
protected = []
def stash(m):
    protected.append(m.group(0))
    return f"\x00{len(protected)-1}\x00"

text = re.sub(r"```.*?```", stash, text, flags=re.S)
text = re.sub(r"`[^`\n]*`", stash, text)
text = re.sub(r"\$\$.*?\$\$", stash, text, flags=re.S)
text = re.sub(r"\$[^$\n]*\$", stash, text)
text = re.sub(r"https?://\S+", stash, text)

# --- (1) remove bold markers (keep inner text); leave single * italics alone ---
text = text.replace("**", "")

# --- (2) em-dash / double-hyphen splices -> comma.
# NOTE: markdown '---' horizontal rules are NOT touched (they separate sections), and
# we never insert a space after an existing comma (that would split numbers like 20,480).
text = re.sub(r"\s*—\s*", ", ", text)                 # em dash (U+2014)
text = re.sub(r"(?<=\S)[ \t]+--[ \t]+(?=\S)", ", ", text)  # spaced double hyphen

# --- tidy punctuation artifacts the replacement can create ---
text = re.sub(r"[ \t]+,", ",", text)          # space before comma
text = re.sub(r",[ \t]*,", ",", text)         # doubled commas
text = re.sub(r",[ \t]*\.", ".", text)        # comma immediately before period
text = re.sub(r":[ \t]*,", ": ", text)        # colon then comma
text = re.sub(r"[ \t]{2,}", " ", text)        # collapse runs of spaces (not newlines)

# --- restore protected spans ---
text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)

open(path, "w", encoding="utf-8").write(text)
print(f"deslopped {path}: removed bold markers; em-dash splices -> commas")
