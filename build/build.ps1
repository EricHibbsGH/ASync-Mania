# build.ps1 — reproducible build of paper/ASync-Mania.pdf from the frozen paper/paper.md.
# Toolchain: pandoc + lualatex (MiKTeX or TeX Live) + Python 3. Run from anywhere:
#   pwsh build/build.ps1
# Content is frozen: this script only reorganizes (move/merge) and renders.
$ErrorActionPreference = "Continue"
$build = $PSScriptRoot
$root  = Split-Path $build -Parent
Set-Location $build

# Resolve pandoc (PATH, else the known user install).
$pandoc = (Get-Command pandoc -EA SilentlyContinue).Source
if (-not $pandoc) { $pandoc = "C:\Users\bojo1\AppData\Local\Pandoc\pandoc.exe" }

# Stage the frozen source + figures next to the build helpers.
Copy-Item (Join-Path $root "paper\paper.md") (Join-Path $build "paper.md") -Force
if (Test-Path (Join-Path $build "figures")) { Remove-Item (Join-Path $build "figures") -Recurse -Force }
Copy-Item (Join-Path $root "figures") (Join-Path $build "figures") -Recurse -Force

# 1) Reorganize into canonical ACM section order (writes paper_canon.md, etc.).
python build_acm.py

# 2) Render single-column PDF (primary deliverable).
$opts = @(
  "--pdf-engine=lualatex","--lua-filter=mathfix.lua","--toc","--toc-depth=3",
  "-V","lof=true","-V","lot=true","-V","documentclass=article","-V","papersize=letter",
  "-V","fontsize=11pt","-V","geometry:margin=1in","-V","colorlinks=true","-V","linkcolor=Blue",
  "-V","urlcolor=Blue","-V","toccolor=Black","-V","linestretch=1.05",
  "--include-in-header=header.tex","--include-before-body=frontmatter.tex"
)
& $pandoc "paper_canon.md" @opts -o (Join-Path $root "paper\ASync-Mania.pdf")

# 3) Mirror the PDF into the Pages site so the embedded viewer resolves.
Copy-Item (Join-Path $root "paper\ASync-Mania.pdf") (Join-Path $root "docs\ASync-Mania.pdf") -Force

if (Test-Path (Join-Path $root "paper\ASync-Mania.pdf")) {
  "BUILD OK -> paper/ASync-Mania.pdf ({0:N0} KB)" -f ([math]::Round((Get-Item (Join-Path $root 'paper\ASync-Mania.pdf')).Length/1KB))
} else { throw "BUILD FAILED: ASync-Mania.pdf not produced" }

# Note: an acmart two-column variant (acm_postprocess.py) is included but is not the primary
# output. Its wide pandoc longtable (Tbl in section 6) and 11 full-width figure floats did not
# place cleanly in two columns on this toolchain; the single-column PDF is the shipped paper.
