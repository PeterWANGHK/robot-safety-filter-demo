# Environment Poisson fields versus robot-centred distance barriers

Both views describe the separation between the same moving robot and the same
obstacles. They do **not** generally produce the same barrier values, gradients,
or filtered commands. A robot-centred display also does not turn a geometric
collision filter into a force, payload or human-injury model.

## 1. The geometric duality

Let `R(q)` be the union of the robot's ten conservative, oriented link boxes and
let `O` be the union of the five static obstacle boxes. When these sets are
disjoint, their Euclidean separation satisfies

\[
\operatorname{dist}(R(q),O)
=\min_{x\in O}\operatorname{dist}(x,R(q))
=\min_{y\in R(q)}\operatorname{dist}(y,O).
\]

Thus an environment field queried by the robot and a robot-centred distance
envelope queried by the environment encode the same *geometric separation*
when distances and complete sets are used consistently. Testing only obstacle
centres, joint centres, or a handful of end-effector points breaks this
equivalence: an object's surface can collide while its centre remains far away.

For a box with centre `c`, orientation `Q` and half sizes `s`, its signed distance
at world point `x` is

\[
u=|Q^T(x-c)|-s,\qquad
d_{\mathrm{box}}(x)=\|\max(u,0)\|_2+\min(\max_j u_j,0).
\]

The minimum over robot boxes is a useful moving envelope. Outside the union it
is the distance to that union. Inside overlapping boxes, the minimum individual
signed distance is not generally the exact signed distance to the union's
outer boundary. It identifies occupied points using a nonpositive value; touching boxes can
also have an internal shared face with zero value. It should be labelled “minimum proxy-box signed distance,” rather than a globally
exact union SDF. Link-focused envelopes are display selections; collision
control must retain all declared robot–obstacle pairs.

## 2. Environment-centred Poisson method

The paper's method solves

\[
\Delta H=-1\quad\text{in buffered free space},\qquad H=0\text{ on its boundary},
\]

then evaluates the field at each moving surface sample `y_i(q)`:

\[
\nabla H(y_i)J_i(q)v\ge-\alpha(H(y_i)-H_{\mathrm{reserve}}).
\]

These are first-order joint-velocity constraints. The implemented `field.py`
uses a fixed known scene and `H_t=0`. Its `H` has units m² and incorporates the
entire free-space boundary, including the outer grid boundary. It is not a
Euclidean clearance field. The paper's surface-cover and obstacle-buffer
construction is explained in `PAPER_AND_MOBILE_LIMITATIONS.md`; its statements
are conditional on coverage, safe initialization, feasible controls and suitable
tracking. The numerical grid and trilinear interpolation do not establish the
paper's smooth continuous-time certificate.

Primary reference: Wilkinson et al., Sections II-B/II-C and III-A–D, equations
(5), (7)–(16), and Theorem 1,
[arXiv v1](https://arxiv.org/html/2604.21189v1).

## 3. Robot-centred nearest-pair method

The comparison uses every link-box/obstacle-box pair, rather than differentiating
one global minimum. Ten robot proxies and five obstacles give **50 pairs**.
For pair `(a,b)`, let

\[
b_{ab}(q)=d_{ab}(q)-d_{\mathrm{margin}}.
\]

For positive separation with a differentiable closest-point solution, let `r`
be the closest point on the robot and `o` the closest point on the fixed
obstacle. Define

\[
n=(r-o)/d_{ab},\qquad
\frac{\partial b_{ab}}{\partial q}=n^T J_r(q).
\]

The Jacobian is evaluated at the world witness point treated as instantaneously
attached to the corresponding robot body. The closest-point envelope theorem
removes the need to differentiate a unique optimizing witness's motion along
the two surfaces. It does not justify choosing an arbitrary derivative at a
nonunique witness or feature switch. The velocity QP uses

\[
n^TJ_r(q)v\ge-\alpha b_{ab}(q)
\]

for **all** pairs, plus the same joint-speed and joint-limit constraints as the
Poisson comparison. A nearest-pair velocity filter is a distinct comparison
method; it is not the paper's Poisson controller translated to another origin.

MuJoCo's `mj_geomDistance(robot_geom, obstacle_geom, distmax, fromto)` returns
the signed distance and, when available, the first geometry's witness followed
by the second's. Consequently `(fromto[:3]-fromto[3:])/d` points from obstacle
to robot. The opposite sign would encourage motion toward the obstacle.
Distances at the query cutoff, absent/nonfinite witnesses, near-zero separation
and unverified degenerate derivatives must not be accepted as certified
constraints. Penetration recovery requires a separate policy and does not
restore a forward-invariance claim from an unsafe initial state.

Primary references:
[MuJoCo distance/Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html),
and the witness assignment in
[official `engine_support.c`, `mj_geomDistance` / `mj_geomDistanceCCD`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_support.c).
Record the installed MuJoCo version and native collision-detector setting
alongside results.

## 4. Why the QPs can disagree

Poisson values depend on global geometry; nearest-pair barriers depend directly
on local metric separation. Even if their zero sets coincide, barrier shape
changes the admissible approach speed. If `H=rho(b)`, equivalent half-spaces
require a correspondingly transformed decay rule,
`alpha_H(rho(b)) = rho'(b) alpha_b(b)`. A positive constant rescaling cancels
for a linear decay rule; arbitrary nonlinear transforms do not. A Poisson
solution generally is not a function of clearance alone.

The practical safe domains also differ. The Poisson implementation adds
surface-cover inflation, voxel allowances, interpolation-cell rejection and an
outer boundary. The pair method checks complete proxy boxes with an explicit
metric margin. Those extra conservatisms must be disclosed rather than
describing unlike parameters as identical safety margins. Their raw minimum
barrier values are in different units and cannot rank safety.

Both methods can stop or become infeasible in clutter, need a nominal
collision-aware planner, and depend on velocity tracking. Box distance and
closest-witness changes are nonsmooth; a single-global-minimum derivative would
add further switching. Neither method currently handles self-collision,
the excluded visual floor, a grasped payload, moving obstacles or mobile-base
stability. Static payload diagnostics remain separate from collision control.

## 5. Verification and comparison protocol

Use the same robot proxies, obstacles, starting pose, target, planned route,
nominal controller, joint limits, velocity caps, plant timestep and stopping
rules. Include both filter-only and planner-plus-filter trials if comparing
reactive blocking. Preserve failed, infeasible and timeout episodes.

Before interpreting trajectories, check:

1. Pair gradient sign and magnitude against central differences at separated,
   stable-witness poses; also test known translation directions and reject
   degenerate/near-contact cases. Preserve the full simulation integration state
   during diagnostics.
2. Agreement of pair distances with independent geometric clearance and robot
   envelope queries for suitably chosen closest witnesses; obstacle-centre-only
   checks are not sufficient. Test touching, overlapping and cutoff cases.
3. QP residuals and joint bounds independently after solving. Re-evaluate actual
   integrated-state clearance; command-side feasibility alone is insufficient.
4. Task success, contacts, minimum whole-body clearance in metres, tracking
   error, corrections, solver failures, planning cost, and mean/tail/full-cycle
   timing. Include Poisson construction time separately from online filtering.
5. Multiple seeds and clutter placements before making general claims. A
   browser animation replays or visualizes generated results; it does not run
   the Python safety controller or validate real hardware.

## 6. Publication provenance

The B601 digital-twin source is pinned locally to
`Yang-Ci/ReBot_Arm_DigitalTwin_DM` commit
`914c763c4adffb5f5744c43ecffbf6199c282bd6`, whose
[license](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM/blob/914c763c4adffb5f5744c43ecffbf6199c282bd6/LICENSE)
is CERN-OHL-W-2.0. Redistributed model assets and modifications retain their
notices, source locations and applicable terms; they should not be relabelled
under the demo's own code license. Avoid copying unrelated ROS/SDK stacks into
the browser publication. Plotly.js and KaTeX use MIT licenses: retain their
copyright and full permission notices. KaTeX assets already carry a pinned
version/hash manifest and embedded notice. Public files should contain only the
explicit demo source, assets and selected synthetic results, with no credentials,
private configuration or machine-specific installation paths.
