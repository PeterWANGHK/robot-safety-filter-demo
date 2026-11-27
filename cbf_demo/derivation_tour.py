"""Compact mathematical guide placed directly beside the parallel 3-D fields.

The tour reads frozen diagnostics and cached velocity projections. It performs
only arithmetic in the browser, renders with the existing offline KaTeX global,
and emits scene-overlay instructions. No geometry/control source is modified.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


_ASSETS = Path(__file__).resolve().parent.parent / 'assets'
_QP = dict(tex=(r'\begin{gathered}v^*=\arg\min_v\tfrac12\|v-v_{\rm nom}\|^2\\'
                r'A_{\rm all}v\ge\beta_{\rm all},\quad\ell(q)\le v\le u(q)\end{gathered}'),
           fallback='v* = closest allowed velocity to vnom; all rows and joint bounds apply')
TOUR_EQUATIONS = {
    'poisson': [
        dict(tex=(r'\begin{gathered}\mathcal O=\bigcup_{j=1}^{5}\mathrm{Box}(c_j,s_j)\\'
                  r'p_i(q)=R_b(q)r_i+t_b(q)\end{gathered}'),
             fallback='O = union of five obstacle boxes; p = R(q) r + t(q)'),
        dict(tex=(r'\begin{gathered}\Delta H=-1,\quad H|_{\partial\Omega}=0\\'
                  r'b_i(q)=H(p_i(q))-H_0\end{gathered}'),
             fallback='ΔH = −1; H = 0 at the domain boundary and obstacles; b = H(p) − H₀'),
        dict(tex=(r'\begin{gathered}\dot p_i=J_i v\\'
                  r'\dot b_i=(\nabla H(p_i)^\top J_i)v=A_i v\end{gathered}'),
             fallback='ṗ = Jv; ḃ = (∇Hᵀ J)v = Av'),
        dict(tex=r'A_i v\ge-\alpha b_i', fallback='A v ≥ −α b'),
        _QP,
    ],
    'distance': [
        dict(tex=r'd_{bo}(q)=\|p_R(q)-p_O(q)\|', fallback='d = ‖pR(q) − pO(q)‖'),
        dict(tex=r'b_{bo}(q)=d_{bo}(q)-\delta', fallback='b = d − δ'),
        dict(tex=(r'\begin{gathered}n=(p_R-p_O)/d_{bo}\\'
                  r'\dot b_{bo}=n^\top J_Rv=A_{bo}v\end{gathered}'),
             fallback='n = (pR − pO)/d; ḃ = nᵀ Jrobot v = Av'),
        dict(tex=r'A_{bo}v\ge-\alpha b_{bo}', fallback='A v ≥ −α b'),
        _QP,
    ],
}


def derivation_tour_fragment():
    """Return the compact guide; place next to #field-comparison-3d.

    Root integration wraps .derivation-scene-area and this aside in a
    .derivation-near-field container. The guide moves first on narrow screens.
    """
    css = (_ASSETS/'derivation_tour.css').read_text(encoding='utf-8')
    return '<style>'+css+'</style>'+'''<aside id="derivation-tour" class="derivation-tour" aria-labelledby="derivation-title">
<header class="dt-heading"><h3 id="derivation-title">From one point to a safer command</h3><span id="derivation-count">1 / 5</span></header>
<div class="dt-modes" role="group" aria-label="Field representation">
<button type="button" data-derivation-mode="poisson" aria-pressed="true">Environment Poisson</button>
<button type="button" data-derivation-mode="distance" aria-pressed="false">Arm distance</button></div>
<label class="dt-region">Follow body region <select id="derivation-region" aria-label="Body region for the mathematical example">
<option value="whole">Whole arm</option><option value="shoulder">Base / shoulder</option>
<option value="elbow" selected>Elbow / forearm</option><option value="wrist">Wrist</option>
<option value="gripper">Gripper</option></select></label>
<nav class="dt-steps" aria-label="Derivation steps">
<button type="button" data-derivation-stage="0" aria-current="step"><span>1</span>Geometry</button>
<button type="button" data-derivation-stage="1"><span>2</span>Barrier</button>
<button type="button" data-derivation-stage="2"><span>3</span>Joint<br>derivative</button>
<button type="button" data-derivation-stage="3"><span>4</span>CBF<br>inequality</button>
<button type="button" data-derivation-stage="4"><span>5</span>Filtered<br>command</button></nav>
<p class="dt-context" id="derivation-context">Loading the selected body point…</p>
<p class="dt-explanation" id="derivation-explanation">Choose a point, then follow its value through the filter.</p>
<div id="derivation-equation" class="math-equation dt-equation" data-tex="p_i(q)=R_b(q)r_i+t_b(q)">p = R(q) r + t(q)</div>
<div id="derivation-numbers" class="dt-numbers" aria-live="polite"></div>
<div id="derivation-symbols" class="dt-symbols" aria-label="Hover or focus for symbol meanings"></div>
<div class="dt-actions"><button id="derivation-previous" type="button" disabled>← Previous</button>
<button id="derivation-next" type="button">Next →</button></div>
<p class="dt-scope" id="derivation-scope">Frozen recorded pose · geometric rates, not forces.</p>
</aside>'''


def derivation_tour_script(data):
    """Return an offline compressed-data mount script, reusing global KaTeX.

    ``data`` is build_derivation_data(..., command_data=...) output. Listeners
    use existing rebot-comparison-change and rebot-comparison-command events.
    They emit rebot-derivation-step with stage 0..4 and scene-overlay geometry.
    """
    payload = dict(diagnostics=data, equations=TOUR_EQUATIONS)
    encoded = base64.b64encode(gzip.compress(
        json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()
    source = (_ASSETS/'derivation_tour.js').read_text(encoding='utf-8')
    return '<script>\n'+source.replace('__DERIVATION_TOUR_GZIP__', encoded)+'\n</script>'
