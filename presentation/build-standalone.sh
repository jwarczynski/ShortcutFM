#!/usr/bin/env bash
# Rebuild a fully self-contained standalone HTML by inlining MathJax.
# Run from anywhere — paths below are absolute relative to the script's directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/index.html"
DST="$HERE/shortcut-flow-matching.standalone.html"
MATHJAX_URL="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MJS_CACHE="$HERE/.mathjax.cache.js"

if [ ! -f "$MJS_CACHE" ]; then
  echo "Downloading MathJax..."
  curl -sLo "$MJS_CACHE" "$MATHJAX_URL"
fi

python3 - <<PY
src = "$SRC"; mjs = "$MJS_CACHE"; dst = "$DST"
with open(src, encoding="utf-8") as f: html = f.read()
with open(mjs, encoding="utf-8") as f: js = f.read()
needle = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" defer></script>'
if needle not in html: raise SystemExit("MathJax tag not found in index.html")
js_safe = js.replace("</script>", r"<\/script>")
out = html.replace(needle, f"<script>{js_safe}</script>", 1)
with open(dst, "w", encoding="utf-8") as f: f.write(out)
import os; print(f"wrote {dst}: {os.path.getsize(dst):,} bytes")
PY
