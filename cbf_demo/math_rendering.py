"""Self-contained, offline KaTeX rendering for generated explanation HTML.

Only the pinned KaTeX distribution is embedded. Its MIT license and upstream
archive/file hashes are retained under assets/katex. Fonts use data URLs; an
HTML report saved elsewhere does not depend on this installation or a network.
"""

from functools import lru_cache
from pathlib import Path
import base64
import re


KATEX_VERSION = "0.16.22"
_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "katex"


@lru_cache(maxsize=1)
def offline_math_assets() -> str:
    """Return inline font-embedded CSS and JavaScript, with no CDN requests."""
    css = (_ASSET_ROOT / "katex.min.css").read_text(encoding="utf-8")
    javascript = (_ASSET_ROOT / "katex.min.js").read_text(encoding="utf-8")
    license_text = (_ASSET_ROOT / "LICENSE").read_text(encoding="utf-8")

    def embed_source(match: re.Match) -> str:
        source = match.group(1)
        woff2 = re.search(r"url\([\"']?(fonts/[^)\"']+\.woff2)[\"']?\)", source)
        if woff2 is None:
            raise RuntimeError("Pinned KaTeX CSS has an unexpected font source")
        relative = Path(woff2.group(1))
        if relative.parent != Path("fonts"):
            raise RuntimeError("Unexpected KaTeX font path")
        encoded = base64.b64encode((_ASSET_ROOT / relative).read_bytes()).decode("ascii")
        # Replace the complete source list, removing .woff/.ttf fallbacks.
        return f'src:url("data:font/woff2;base64,{encoded}") format("woff2")'

    css = re.sub(r"\bsrc:([^;}]+)", embed_source, css)
    for reference in re.findall(r"url\(([^)]+)\)", css):
        if not reference.strip("\"'").startswith("data:font/woff2;base64,"):
            raise RuntimeError("KaTeX CSS still contains a non-embedded resource")
    if "http:" in css or "https:" in css or "fonts/" in css:
        raise RuntimeError("KaTeX CSS contains an external resource reference")
    javascript = re.sub(r"</script", r"<\\/script", javascript, flags=re.IGNORECASE)
    return (
        f'<!-- KaTeX {KATEX_VERSION}, MIT license; vendored provenance: assets/katex/metadata.json -->\n'
        f'<!--\n{license_text.replace("--", "—")}\n-->\n'
        f'<style data-katex-version="{KATEX_VERSION}">{css}</style>\n'
        f'<script data-katex-version="{KATEX_VERSION}">{javascript}</script>\n'
    )


def math_render_script() -> str:
    """Render data-tex elements safely, preserving text on missing/error cases.

    HTML attribute entities are decoded by the browser before dataset access.
    KaTeX's untrusted commands stay disabled. Parse errors restore the existing
    textual fallback and log one warning, rather than hiding the explanation.
    """
    return r'''<script>
(function () {
  "use strict";
  function renderMath() {
    if (!window.katex || typeof window.katex.render !== "function") {
      console.warn("Offline math renderer unavailable; preserving equation text.");
      return;
    }
    document.querySelectorAll(".math-equation[data-tex]").forEach(function (element) {
      if (element.dataset.mathRendered === "true") return;
      var fallback = element.textContent;
      try {
        window.katex.render(element.dataset.tex, element, {
          throwOnError: false,
          output: "htmlAndMathml",
          trust: false,
          displayMode: true
        });
        if (element.querySelector(".katex-error")) {
          throw new Error("KaTeX rejected this expression");
        }
        element.dataset.mathRendered = "true";
      } catch (error) {
        element.textContent = fallback;
        element.dataset.mathRendered = "false";
        console.warn("Equation rendering failed; preserving equation text.", error);
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMath, {once: true});
  } else {
    renderMath();
  }
}());
</script>
'''
