-- mathfix.lua — FORMAT-ONLY: inside math spans only, turn the markdown-escaped
-- asterisk "\*" back into a literal "*" so superscript-star equations (k^*, \sigma^*)
-- typeset correctly under LaTeX. paper.md wrote "k^\*" to stop Markdown from reading
-- the asterisk as italics; in $...$ pandoc passes it verbatim and LaTeX rejects "\*".
-- This touches MATH ONLY (prose escapes are left alone) and changes no equation meaning:
-- k^\*  ->  k^*  (i.e. "k-star").
function Math(m)
  m.text = m.text:gsub("\\%*", "*")
  return m
end
