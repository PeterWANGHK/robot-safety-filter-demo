"""Beginner-readable HTML explaining the implemented field and load analysis.

``math_explanation_html(metadata=None)`` returns a self-contained semantic
fragment. Optional metadata accepts the saved configuration.json structure;
otherwise the documented default scene is used. Equation elements have an
escaped ``data-tex`` attribute plus readable Unicode text. A host can replace
their contents with a local math renderer without needing a network service.
The fragment itself starts no renderer, simulator, ROS node or hardware.
"""

from __future__ import annotations

from html import escape
import math

from .scenes import FIELD_ORIGIN, FIELD_SHAPE, FIELD_SPACING, INFLATION, clutter_scene


def _equation(label: str, tex: str, fallback: str) -> str:
    return (
        '<figure class="math-block"><figcaption>' + escape(label) + '</figcaption>'
        '<div class="math-equation" data-tex="' + escape(tex, quote=True) + '">'
        + escape(fallback) + '</div></figure>'
    )


def _vector(values) -> str:
    return '(' + ', '.join(f'{float(value):.4g}' for value in values) + ')'


def math_explanation_html(metadata=None) -> str:
    """Return an embeddable mathematical guide for this B601 experiment.

    Values supplied in optional metadata should describe the displayed run.
    Recognized saved-run keys are grid_shape, spacing_m,
    requested_inflation_m and model.{proxy_count,surface_point_count,
    coverage_radius_m}. The field origin and obstacle scene are this module's
    current synthetic scene; the text identifies control numbers as defaults.
    """
    metadata = {} if metadata is None else metadata
    model = metadata.get('model', {})
    shape = tuple(int(value) for value in metadata.get('grid_shape', FIELD_SHAPE))
    spacing = float(metadata.get('spacing_m', FIELD_SPACING))
    inflation = float(metadata.get('requested_inflation_m', INFLATION))
    origin = tuple(float(value) for value in FIELD_ORIGIN)
    upper = tuple(origin[axis] + spacing * (shape[axis]-1) for axis in range(3))
    count = int(model.get('surface_point_count', 798))
    body_count = int(model.get('proxy_count', 10))
    coverage = float(model.get('coverage_radius_m', 0.0229316383308848))
    voxel_allowance = math.sqrt(3)*spacing/2
    collar = math.sqrt(3)*spacing
    obstacle_rows = []
    for obstacle in clutter_scene():
        center, half = obstacle['center'], obstacle['half_size']
        lower = [center[axis]-half[axis] for axis in range(3)]
        higher = [center[axis]+half[axis] for axis in range(3)]
        obstacle_rows.append(
            '<tr><th scope="row">' + escape(obstacle['name'].replace('_', ' ')) + '</th>'
            '<td>' + _vector(center) + '</td><td>' + _vector(half) + '</td>'
            '<td>' + _vector(lower) + ' → ' + _vector(higher) + '</td></tr>'
        )

    eq = _equation
    return f'''<section class="math-explanation" id="math-guide" aria-labelledby="math-guide-title">
<style>
.math-explanation {{ color: #18324a; line-height: 1.65; max-width: 1100px; margin: 1.5rem auto; }}
.math-explanation details {{ border: 1px solid #d8e1e8; border-radius: 10px; margin: .8rem 0; padding: .8rem 1rem; background: #fff; }}
.math-explanation summary {{ cursor: pointer; font-size: 1.06rem; font-weight: 650; color: #173c5a; }}
.math-explanation .math-block {{ margin: 1rem 0; padding: .8rem; background: #f3f7fa; border-left: 3px solid #0072b2; overflow-x: auto; }}
.math-explanation figcaption {{ font-size: .86rem; color: #4e6578; margin-bottom: .3rem; }}
.math-explanation .math-equation {{ font-family: ui-serif, Georgia, serif; white-space: pre-wrap; }}
.math-explanation table {{ border-collapse: collapse; width: 100%; font-size: .91rem; }}
.math-explanation th,.math-explanation td {{ border-bottom: 1px solid #dce4ea; padding: .5rem; text-align: left; vertical-align: top; }}
.math-explanation .math-scope {{ padding: .75rem 1rem; background: #edf5fa; border-radius: 8px; }}
.math-explanation .table-scroll {{ overflow-x: auto; }}
</style>
<h2 id="math-guide-title">How the 3D field changes the arm's motion—and what load means</h2>
<p>The colored volume is a scalar function of <strong>position in the room</strong>.
The controller evaluates it on the moving surfaces of the arm, converts each
spatial gradient into a joint-velocity constraint, and changes the requested
motion only as much as needed. The load calculation is a separate static torque
analysis; a bright field color is not a force or a motor-capacity measurement.</p>
<p class="math-scope">Current scope: fixed-base arm, synthetic static boxes,
open fixed gripper, and a first-order velocity CBF. The load panel estimates a
fixed-pose holding requirement; it does not attach a simulated payload to the
dynamic experiment or regulate contact force.</p>

<details open><summary>1. Where the surrounding regions come from</summary>
<p>The field spans <strong>{_vector(origin)} to {_vector(upper)} m</strong> in the
base frame, on a <strong>{' × '.join(map(str, shape))}</strong> grid with
<strong>{spacing:g} m</strong> spacing. In the default run this is
[-0.8, 0.8] × [-0.8, 0.8] × [-0.15, 1.25] m. It contains five physical boxes:</p>
{eq('Physical occupied region: a union of world-aligned boxes', r'\mathcal O_j=\prod_{a=1}^{3}[c_{j,a}-s_{j,a},c_{j,a}+s_{j,a}],\qquad \mathcal O=\bigcup_{j=1}^{5}\mathcal O_j', 'Each obstacle is the product of its three coordinate intervals; the occupied environment is the union of the five boxes.')}
<div class="table-scroll"><table><thead><tr><th>Obstacle</th><th>Center XYZ (m)</th><th>Half-size XYZ (m)</th><th>Lower → upper XYZ (m)</th></tr></thead>
<tbody>{''.join(obstacle_rows)}</tbody></table></div>
<p>A grid voxel is marked occupied when its spacing-wide, world-aligned cube
intersects a physical box. The occupied grid-site mask is then expanded by
{inflation:.5f} m plus a half-voxel diagonal ({voxel_allowance:.5f} m here).
The requested {inflation:.3f} m includes the default 0.025 m surface-cover
reserve and 0.010 m extra model/tracking allowance. This allowance is a chosen
margin, not a proven bound on all physical tracking errors.</p>
<p>A further 3 × 3 × 3 grid-neighborhood collar is used by the PDE solver.
Its largest additional corner displacement is {collar:.5f} m. The voxel
intersection, discrete expansion, collar and rejected interpolation cells are
different geometric operations: <strong>their sum is not an exact uniform
spherical clearance radius</strong>. The collar makes the interpolated field
approach zero before the validity check rejects a cell.</p>
{eq('Poisson field used in this implementation', r'\Delta h(p)=-1\quad(p\in\Omega_{\mathrm{solve}}),\qquad h=0\quad\text{on obstacle and outer-domain boundaries}', '∇²h(p) = −1 in the solver free domain; h = 0 on its obstacle and outer-box boundaries.')}
<p>Negative constant forcing produces positive interior values. With this
choice, h has units <strong>m²</strong>. A seven-point finite-difference solve
computes the grid; trilinear interpolation provides h at arbitrary points and
its exact within-cell derivative. Its gradient can jump at voxel faces, so this
discretization does not inherit a smooth continuous-time safety theorem.</p>
{eq('Seven-point finite-difference equation at a free interior grid site', r'\sum_{a=1}^{3}\frac{h_{g+e_a}-2h_g+h_{g-e_a}}{\Delta^2}=-1,\qquad \Delta=0.025\ \mathrm m\ \text{by default}', 'Sum of the three second coordinate differences divided by grid-spacing squared equals −1; occupied and boundary sites have h = 0.')}
<p>An isosurface is simply the set h(p) = c. The orange and blue shells are
different values of the same function, not individual obstacles or force
fields. The display crop opens some shells for visibility; it changes neither
the PDE domain nor the controller. Low h can also occur near the outer box.
Whole-box clearance is measured separately, in metres.</p></details>

<details><summary>2. Shoulder, elbow and wrist: evaluate the surfaces, not just the joint centers</summary>
<p>The visual meshes are enclosed by <strong>{body_count} body-local proxy boxes</strong>.
They are sampled at <strong>{count} surface points</strong>; the default proven
face-grid covering radius is at most <strong>{coverage:.7f} m</strong>. This
means every point on a proxy face lies within that radius of a retained sample.
The boxes can be larger than the real links and therefore block narrow gaps.</p>
{eq('Move one sample from its link frame into the room', r'p_{b,k}(q)=R_b(q)r_{b,k}+t_b(q),\qquad J_{b,k}(q)=\frac{\partial p_{b,k}}{\partial q}', 'p[b,k](q) = R[b](q) r[b,k] + t[b](q); J[b,k] = ∂p[b,k]/∂q.')}
<p>Here b names a rigid body, k names its local sample, and q contains the six
joint angles. Forward kinematics supplies its rotation R and translation t.
An elbow sample moves when several upstream joints move; its Jacobian therefore
usually has several nonzero columns. The same reasoning applies to shoulder,
wrist and finger samples.</p>
{eq('Individual barriers and a useful body-summary value', r'h_{b,k}(q)=h(p_{b,k}(q)),\qquad h_{b,\min}(q)=\min_k h_{b,k}(q)', 'h[b,k](q) = h(p[b,k](q)); body summary h[b,min](q) = min over its surface samples.')}
<p>A body's minimum is useful for coloring and inspection. Within the valid
interpolated domain it is continuous, but it can have a kink when the critical
sample changes; voxel-face gradients can also jump. The practical controller
keeps <strong>one constraint per retained surface point</strong> instead of
differentiating a single minimum or checking only joint centers.</p>
<p>“Shoulder” and “elbow” are engineering labels here. J1/J2 are the proximal
group, J3 the elbow group, and J4–J6 the wrist group; these are not human anatomical
models. A graph labeled for one joint can show that coordinate's contribution
to many surface constraints, rather than an independent barrier attached only
to that joint.</p>
<p>The region selector is a <strong>visual focus only</strong>: base + shoulder
contains base_link/link1/link2; elbow/forearm contains link3; wrist contains
link4/link5/link6; gripper contains end_link and both fingers. All {count}
surface constraints remain active regardless of the displayed selection.</p></details>

<details><summary>3. The exact velocity-CBF constraint and feedback loop</summary>
{eq('Chain rule from the 3D field to joint motion', r'\dot h_i=\nabla h(p_i)^T J_i(q)v+\partial_t h_i,\qquad A_i=\nabla h(p_i)^T J_i(q)', 'ḣ[i] = ∇h(p[i])ᵀ J[i](q) v + ∂h[i]/∂t; A[i] = ∇h(p[i])ᵀ J[i](q).')}
<p>The paper uses direct joint-velocity control, q̇ = v. Our default field is
static, so its time derivative is zero. The code adds a small positive reserve
h₀ = 2 × 10⁻⁵ m². With gain α = 2 s⁻¹, each point imposes:</p>
{eq('Actual constraint, including an optional tracking-error bound', r'A_i v\ge -\alpha(h_i-h_0)-\partial_t h_i+d_v\lVert\nabla h(p_i)\rVert,\qquad d_v=0\ \text{by default}', 'A[i] v ≥ −α(h[i] − h₀) − ∂h[i]/∂t + dᵥ ‖∇h(p[i])‖; default dᵥ = 0.')}
<p>If h is large, approach is allowed. Near the reserve, motion that decreases
h is restricted, while motion along a safe direction can remain possible.
The optional dᵥ term would require a justified Cartesian tracking-error bound;
the default does not establish one merely by logging an observed error.</p>
<p>For a revolute coordinate, ∂h/∂q has units m²/rad, and multiplying it by
rad/s gives m²/s. This joint sensitivity is separate from motor torque in N·m.</p>
{eq('Smallest correction to the requested velocity', r'v^*=\arg\min_v\frac12\lVert v-v_{\mathrm{nom}}\rVert^2\quad\text{subject to all point constraints and }\ell(q)\le v\le u(q)', 'Choose v* to minimize ½‖v − v[nominal]‖², subject to every surface CBF and the joint bounds.')}
{eq('Default speed and joint-limit barriers', r'\ell_j=\max[-v_{\max},-\alpha(q_j-q_{\min,j}-\delta_q)],\quad u_j=\min[v_{\max},\alpha(q_{\max,j}-\delta_q-q_j)]', 'lower[j] = max(−vmax, −α(q[j]−qmin[j]−δq)); upper[j] = min(vmax, α(qmax[j]−δq−q[j])).')}
<p>Defaults are vₘₐₓ = 0.45 rad/s and δq = 0.015 rad. Rows are normalized
numerically; only rows proven redundant over the entire allowed velocity box
are omitted from the solve. OSQP is the primary solver; a logged SLSQP fallback
solves the identical convex QP if needed. Every original constraint and bound
is checked afterward. There is no added safety slack or clipping of accepted v.</p>
<p>The hierarchy supplies a nominal route: <strong>sampled reachability →
collision-aware RRT planner → pointwise CBF filter → saturated PD+bias joint
servo</strong>. The filter uses actual simulated q and q̇. Torque saturation
can prevent accurate tracking even if this velocity QP is feasible. Invalid
field queries or an unsuccessful check trigger an abort; zero requested velocity
does not prove safe braking for an inertial physical arm.</p></details>

<details><summary>4. Can these colors be interpreted as repulsive forces?</summary>
<p>The field and its gradient are geometric quantities: h is in m² and ∇h in
metres. They are <strong>not measured newtons</strong>. The CBF changes joint
velocity; the physical model's servo then generates the applied torques.</p>
<p>For intuition only, one could choose an artificial energy scale κ in joules
and a reference h꜀, and define the following virtual potential:</p>
{eq('One optional explanatory potential—not used by this controller', r'U(p)=\frac{\kappa}{2}\left[\max\left(0,1-\frac{h(p)}{h_c}\right)\right]^2,\quad F_{\mathrm{virtual}}=-\nabla_pU,\quad\tau_{\mathrm{virtual}}=J^TF_{\mathrm{virtual}}=-\nabla_q U', 'U = ½κ[max(0,1−h/hc)]²; Fvirtual = −∇pU; τvirtual = JᵀFvirtual = −∇qU.')}
<p>Its arrows would point toward increasing h where the potential is active.
Changing the arbitrary κ changes their magnitude without changing the scene.
Even if units are chosen to make the arrows read in newtons, they remain a
designed virtual effect, not contact-force feedback, a prediction of collision
impact or the actual CBF correction. The Jacobian-transpose relation follows
from matching endpoint and joint mechanical power, as derived in
<a href="https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/">Modern Robotics, §5.2</a>.</p></details>

<details><summary>5. How a held object changes motor torque and headroom</summary>
<p>The static panel places a hypothetical point-mass payload at the tool COM
position, with an optional offset fixed in the tool frame. At a fixed pose and
zero velocity, the arm's own model supplies its gravity-compensation torque G(q).
Use the world-down vector g⃗ = (0, 0, −9.81) m/s²:</p>
{eq('The offset follows the tool, not the room', r'p_{\mathrm{COM}}(q)=p_{\mathrm{TCP}}(q)+R_{\mathrm{TCP}}(q)r_{\mathrm{COM}},\quad r_{\mathrm{COM}}=d e_a,\quad a\in\{x,y,z\},\quad |d|\le0.20\ \mathrm m', 'pCOM = pTCP + RTCP · rCOM; choose tool x, y or z and an offset between −0.20 and +0.20 m.')}
{eq('Static holding demand and its sign convention', r'F_L=m\mathbf g,\quad \tau_{\mathrm{external}}=J_{\mathrm{COM}}^T F_L,\quad \tau_{\mathrm{hold}}(q,m)=G(q)-J_{\mathrm{COM}}^T m\mathbf g=G(q)+m a(q)', 'Payload force F = m g⃗; external joint torque = JCOMᵀF; required motor torque τhold = G − JCOMᵀm g⃗ = G + m a.')}
<p>The minus sign means the motors oppose the external downward load.
For an offset COM, use the Jacobian <em>at that COM</em>, or use a TCP wrench
with moment r × F once; applying both would double-count the lever arm.
A 1 kg object with a 0.30 m perpendicular lever arm contributes 2.94 N·m
about that axis, before the arm's own weight. Other axes can have very different
demands, including zero torque when the force line passes through an axis.</p>
{eq('Deterministic utilization and signed feasible mass interval', r'c_j=\rho\tau_{\lim,j},\quad u_j=\frac{|\tau_{\mathrm{hold},j}|}{c_j},\quad m\ge0,\quad -c_j\le G_j+m a_j\le c_j\ \ \forall j', 'Chosen cap c[j] = ρ τlimit[j]; utilization u[j] = |τhold[j]|/c[j]; intersect every signed inequality −c[j] ≤ G[j]+m a[j] ≤ c[j], with m ≥ 0.')}
<p>The model uses URDF limits (27, 27, 27, 7, 7, 7) N·m and a default
headroom fraction ρ = 0.7 for this screening calculation. These are not
validated continuous motor ratings. The currently most loaded axis has the
largest utilization; the axis that limits a later increase in payload mass can
be different. Capacity is not obtained by adding six motor torques.</p>
<p>The signed intersection matters: an added load can counterbalance gravity
on one axis. If the unloaded pose already exceeds the chosen caps but a positive
mass interval remains, the panel labels it a <em>counterweight-only interval</em>,
not a safe payload recommendation. COM offset and pose can change the interval.</p>
<p>The interactive 0–4 kg slider intentionally explores model stress cases.
It is not a hardware payload range. Seeed lists <strong>1.5 kg, without gripper,
with 70% reach recommended</strong> on its
<a href="https://www.seeedstudio.com/reBot-Arm-B601-DM-p-6740.html">B601-DM product page</a>.
A larger torque-only model interval does not supersede that qualified specification.</p>
{eq('A useful conceptual intersection—not an implemented torque filter', r'\mathcal C_{\mathrm{geom}}=\{q:\ h_i(q)\ge h_0\ \forall i,\ \text{valid domain}\},\quad\mathcal C_{\mathrm{load}}=\{q:\ |\tau_{\mathrm{hold},j}|\le c_j\ \forall j\},\quad\mathcal C_{\mathrm{screen}}=\mathcal C_{\mathrm{geom}}\cap\mathcal C_{\mathrm{load}}', 'Conceptually retain poses passing both geometric constraints and static torque caps: Cscreen = Cgeom ∩ Cload.')}
{eq('Load-dependent positional reachability is a projection of feasible configurations', r'\mathcal R(m)=\{p_{\mathrm{TCP}}(q):q\in\mathcal Q\cap\mathcal C_{\mathrm{screen}}(m)\}', 'R(m) is the set of tool positions produced by joint-limit-valid configurations passing both geometry and static load constraints.')}
<p>Different elbow arrangements can reach the same tool position with different torque demand and collision clearance. A position-only reach map hides these configurations; orientation, grasp and connecting-path requirements further restrict reachability.</p>
<p>This intersection is a static screening idea. The load-panel controls do not
add torque constraints to the running velocity CBF or change the simulation's
motor limits.</p></details>

<details><summary>6. What “failure” and a human comparison can—and cannot—mean</summary>
<div class="table-scroll"><table><thead><tr><th>Observation</th><th>Meaning in this model</th></tr></thead><tbody>
<tr><th scope="row">Utilization above 1</th><td>Static demand exceeds a selected deterministic torque cap; it is not a measured probability of damage.</td></tr>
<tr><th scope="row">Utilization below 1</th><td>The screened pose satisfies those model inequalities; thermal limits, gearing, grasp/friction, payload inertia, contact impacts and structural strength still need validation.</td></tr>
<tr><th scope="row">CBF QP feasible</th><td>A commanded joint velocity satisfies the modeled geometric constraints; overloaded motors may still fail to track it.</td></tr>
<tr><th scope="row">Solver iteration limit</th><td>A numerical convergence failure, distinct from a proof of infeasibility or a geometric deadlock.</td></tr>
<tr><th scope="row">Progress stall</th><td>An operational trigger to replan, using a logged progress rule; it does not prove every alternative route is blocked.</td></tr>
</tbody></table></div>
<p>A failure probability needs calibrated distributions for loads, model errors,
strength and operating conditions, plus repeated data. This panel supplies
none of those distributions and therefore reports utilization/headroom only.</p>
<p>A human arm also balances loads through joint moments and lever arms, but
robot motor torque does not translate directly into muscle force. Muscle moment
arms, multiple agonists and antagonists, co-contraction, posture, fatigue and
individual physiology are unknown here. Shoulder/elbow labels support a
mechanical analogy; they do not make this a human strength or injury model.</p></details>

<details><summary>7. What changes for a torque-level CBF or a mobile manipulator?</summary>
{eq('Dynamics needed for an actuator-aware extension', r'M(q)\ddot q+C(q,\dot q)\dot q+G(q)=\tau+J^TF_{\mathrm{ext}},\quad \ddot b+(k_1+k_2)\dot b+k_1k_2 b\ge0', 'M(q) q̈ + C(q,q̇) q̇ + G(q) = τ + JᵀFexternal; a relative-degree-two condition can impose b̈ + (k1+k2)ḃ + k1k2b ≥ 0.')}
{eq('Static environment: torque enters through the second derivative', r'\dot b=\nabla_q b^T\dot q,\qquad \ddot b=\dot q^T\nabla_q^2 b\,\dot q+\nabla_q b^TM^{-1}(\tau-C\dot q-G+J^TF_{\mathrm{ext}})', 'ḃ = ∇q bᵀ q̇; b̈ = q̇ᵀ Hessian(b) q̇ + ∇q bᵀ M⁻¹(τ − Cq̇ − G + JᵀFext).')}
<p>The geometric constraint then becomes linear in commanded torque when the current state and dynamics are known. A moving payload must be included consistently in M, C and G (and its collision geometry); do not also count its gravity as a second external load.</p>
{eq('Initial conditions required by this exponential HOCBF construction', r'b(q_0)\ge0,\qquad \psi_1(q_0,\dot q_0)=\dot b(q_0,\dot q_0)+k_1b(q_0)\ge0', 'Initially b ≥ 0 and ψ1 = ḃ + k1 b ≥ 0; the torque constraints must also be feasible.')}
<p>For b(q,t) = h(p(q),t), a torque-level filter must include inertia, the
field Hessian, Jacobian derivatives and any time derivatives, enforce torque
limits and start inside the appropriate position-and-velocity safe set.
The current first-order filter and static load panel do not implement this.</p>
<p>A wheeled base adds position and heading, base/torso geometry, localization
uncertainty, wheel constraints, braking and tipping limits. Its world-point
Jacobians must include base motion; a differential-drive base cannot command
independent sideways velocity. The same hierarchy remains useful, with a new
whole-body model and validation rather than only more field samples.</p>
<p>The starting method is described in
<a href="https://arxiv.org/html/2604.21189v1">Wilkinson et al., §§II–III</a>.
The finite grid, constant forcing, proxy boxes, solver fallback, static load
screen and hierarchy above are explicit choices of this preliminary reproduction.</p>
</details></section>'''
