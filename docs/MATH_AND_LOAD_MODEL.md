# Mathematics of the B601 safety field and static load screen

The colored surrounding regions are values of a spatial function. The control
filter evaluates that function on moving arm surfaces and constrains joint
velocity. Static load analysis is a separate calculation of holding torque.
Field color is neither force nor motor capacity.

This document describes the default fixed-base experiment. It uses known,
synthetic obstacles and an open, fixed gripper. The static payload screen does
not attach a payload to the dynamic simulator or implement contact-force control.

## 1. Build the surrounding environment field

The grid origin is `(-0.8, -0.8, -0.15) m`, its shape is `(65,65,57)` and its
spacing is `0.025 m`. Its outer bounds are therefore

\[
[-0.8,0.8]\times[-0.8,0.8]\times[-0.15,1.25]\ \mathrm m.
\]

The five physical axis-aligned boxes are:

| Obstacle | Center XYZ (m) | Half-size XYZ (m) |
|---|---|---|
| Central post | (0.25, 0, 0.48) | (0.04, 0.04, 0.19) |
| Table | (0.50, 0, 0.06) | (0.30, 0.40, 0.035) |
| Far post | (0.48, 0.35, 0.40) | (0.035, 0.035, 0.28) |
| Side crate | (0.35, −0.43, 0.25) | (0.12, 0.04, 0.12) |
| Rear crate | (−0.40, 0.35, 0.28) | (0.09, 0.06, 0.15) |

For a box with center `c` and half-size `s`, the physical region is
`c−s <= p <= c+s` coordinate by coordinate. A grid voxel is marked occupied
when its spacing-wide cube intersects a physical box. This avoids representing
a thin box only when it happens to contain a grid center.

The required occupied-site mask is expanded by the requested **0.035 m** plus
a half-voxel diagonal, **0.0216506 m**. The requested 0.035 m comprises a 0.025 m
surface-cover reserve and a chosen 0.010 m extra model/tracking allowance. The
latter is not a proven bound on all physical tracking errors.

The solver mask adds a full 3×3×3 neighborhood around that mask. Its maximum
additional corner displacement is **0.0433013 m**. A rejected interpolation cell
touches the required occupied mask. The extra solver collar makes all eight
corners of every such cell zero, so the interpolated field tends to zero before
a trajectory enters an invalid cell. This corrects an otherwise invisible
positive-value-to-invalid jump.

These are voxel intersection, discrete Euclidean expansion, neighborhood
dilation and interpolation-cell tests. Their numerical distances should **not**
be added and presented as an exact, uniform spherical stand-off distance.
The enlarged regions are deliberately conservative and may close narrow gaps.

The field solves

\[
\Delta h(p)=-1\quad\text{in the solver free domain},\qquad
h=0\quad\text{on obstacle and outer-grid boundaries}.
\]

Negative constant forcing yields positive interior values. With this forcing,
`h` has units **m²**, rather than metres. The implementation uses a seven-point
finite-difference Laplacian, conjugate gradients and a discrete-sine-transform
preconditioner. It reports convergence, residual and setup time.

At arbitrary points, trilinear interpolation supplies `h` and the exact
derivative of the interpolated value. The derivative can jump at voxel faces.
Invalid or occupied-cell queries are rejected independently; zero or positive
scalar values alone are insufficient to establish valid free space.

An isosurface `h(p)=c` shows one level of this single field. Orange and blue
shells are not separate obstacles or physical force layers. Cropping removes
display triangles only and does not alter the PDE or control constraints.
Low `h` also occurs near the outer grid boundary. Minimum whole-box clearance
is calculated separately in metres.

## 2. Evaluate the field on shoulder, elbow and wrist surfaces

The model encloses the visual meshes with ten rigid body-local proxy boxes:
base, six arm links, end link and two fingers. Their default face grids contain
798 samples. The calculated face-cover radius is at most **0.0229316383 m**:
every point on every proxy face lies within that distance of some sample.
This covers the declared boxes, rather than certifying unmodeled hardware.

Let `r_bk` be a sample fixed in body `b`. Forward kinematics gives

\[
p_{b,k}(q)=R_b(q)r_{b,k}+t_b(q),\qquad
J_{b,k}(q)=\frac{\partial p_{b,k}}{\partial q}.
\]

`p_bk` is its position in the room. Its Jacobian has contributions from the
upstream joints that can move it. A distal elbow or wrist sample therefore
usually depends on more than one joint coordinate.

The composed barrier for that sample is

\[
h_{b,k}(q)=h(p_{b,k}(q)).
\]

A useful display summary is `h_b,min(q)=min_k h_bk(q)`. On the valid interpolated
free domain this finite minimum is continuous, but it can have a kink when the
critical sample changes. Voxel-face gradients add another nonsmoothness.
The controller therefore retains one constraint per sample, instead of
differentiating a single body minimum or checking only joint centers.

The engineering groups J1/J2 (proximal/shoulder), J3 (elbow), and J4–J6 (wrist)
are descriptive labels, not human anatomy. A joint's influence is the
corresponding column of each point Jacobian. There is no independent scalar
barrier obtained merely by evaluating the field at a joint angle.

The interactive region selector changes visual focus only: base + shoulder
contains base_link/link1/link2; elbow/forearm contains link3; wrist contains
link4/link5/link6; gripper contains end_link and both fingers. All 798 point
constraints remain active regardless of which region is shown.

## 3. Convert each field sample into a velocity constraint

The paper's kinematic model is `qdot=v`; it uses a high-DOF arm with a
**first-order** velocity CBF. It is not a higher-relative-degree torque
controller. Its chain-rule construction is described in
[Wilkinson et al., §§II–III](https://arxiv.org/html/2604.21189v1).

For each point `i`:

\[
\dot h_i=\nabla h(p_i)^T J_i(q)v+\partial_t h_i,
\qquad A_i=\nabla h(p_i)^T J_i(q).
\]

For a revolute coordinate, `dH/dq` has units m²/rad. Multiplying by joint
velocity in rad/s gives a field rate in m²/s. This geometric sensitivity is
independent of motor torque in N·m.

The implementation imposes

\[
A_i v\ge-\alpha(h_i-h_0)-\partial_t h_i
+d_v\lVert\nabla h(p_i)\rVert.
\]

Defaults are `alpha=2 s⁻¹`, `h0=2e−5 m²`, static `h_t=0`, and `d_v=0`.
`d_v` is an optional bound on Cartesian sample velocity tracking error; the
default does not certify one by logging observed tracking errors.
Near the reserve, approach in a direction that decreases `h` is restricted.
The joint velocity is selected by

\[
v^*=\arg\min_v\frac12\lVert v-v_{nom}\rVert^2,
\quad\text{subject to every point constraint and }\ell(q)\le v\le u(q).
\]

The joint bounds are

\[
\ell_j=\max[-v_{max},-\alpha(q_j-q_{min,j}-\delta_q)],
\quad
u_j=\min[v_{max},\alpha(q_{max,j}-\delta_q-q_j)],
\]

with `vmax=0.45 rad/s` and `delta_q=0.015 rad` by default. The implementation
normalizes rows and omits only constraints proven redundant over the entire
allowed velocity box. It independently checks every original row and bound
after solving. OSQP is primary; a logged SLSQP fallback solves the identical
convex problem when needed. There is no safety slack or post-solve clipping of
accepted velocities.

The hierarchy is:

1. Sampled position reachability supplies a joint seed; it is not complete IK.
2. RRT-Connect finds a collision-checked route at finite edge resolution.
3. The pointwise CBF minimally changes its nominal velocity.
4. Actual simulated `q/qd` feed a saturated PD+bias servo.

The servo applies

\[
\tau=\operatorname{clip}_{\tau_{lim}}
\{K_p(q_{ref}-q)+K_d(v_{ref}-\dot q)+b(q,\dot q)\}.
\]

Velocity-QP feasibility does not prove a torque-limited plant can track its
solution. Field invalidity, failed optimization checks and contact are reported
as failures. A request for zero velocity is not a proof of safe inertial braking.
The discrete interpolation and sampled-time monitor do not inherit a smooth
continuous-time safety theorem.

## 4. Explain repulsion without inventing contact forces

`h` and its gradient have units m² and m. Neither directly measures newtons.
The actual CBF changes velocity; the servo subsequently produces actuator torque.

For intuition, a separate virtual potential could be defined using a chosen
energy scale `kappa` in joules and reference level `h_c`:

\[
U(p)=\frac{\kappa}{2}
\left[\max\left(0,1-\frac{h(p)}{h_c}\right)\right]^2,
\quad
F_{virtual}=-\nabla_p U,
\quad
\tau_{virtual}=J^TF_{virtual}=-\nabla_qU.
\]

This illustrative potential is **not used by the implemented controller**.
Its active arrows point toward increasing `h`, but arbitrary `kappa` changes
their magnitude without changing geometry. Assigning consistent physical units
does not make these arrows measured contact forces, predicted impact loads or
the actual CBF correction.

The Jacobian-transpose relationship follows by equating endpoint mechanical
power `FᵀJ qdot` with joint power `tauᵀqdot`; see the derivation in
[Modern Robotics, §5.2](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/).

## 5. Quantify a hypothetical static payload

The load screen evaluates a fixed pose at zero joint velocity. It uses the
arm's own model gravity compensation `G(q)` and a hypothetical payload mass
`m` at an optionally offset tool COM. It does not step payload dynamics.

The interactive offset is fixed in a selected tool x, y or z direction: r_COM = d e_a with |d| ≤ 0.20 m. It rotates with the tool:

\[
p_{COM}(q)=p_{TCP}(q)+R_{TCP}(q)r_{COM}.
\]

It rotates with the tool rather than staying parallel to world vertical.

Define the world gravity vector `gvec=(0,0,−9.81) m/s²`. With downward external
force `F_L=m*gvec`, the external generalized torque is `J_COMᵀF_L`. The motor
torque needed to oppose it is

\[
\tau_{hold}(q,m)=G(q)-J_{COM}(q)^T m\mathbf g
=G(q)+m a(q),\qquad a(q)=-J_{COM}^T\mathbf g.
\]

The minus sign opposes the applied external force. `a` is the required motor
holding increment per kilogram. For an offset, use the Jacobian at the COM or
the TCP wrench containing the moment `r×F` exactly once; using both would
double-count the lever arm.

As a scale example, 1 kg acting with a 0.30 m perpendicular lever arm produces
2.94 N·m about that axis, before accounting for the robot's own weight. The
torque about another axis can be zero if the load line passes through it.
J1 is a vertical yaw axis in this fixed-base model, so a purely gravitational
payload does not create static J1 holding torque. That says nothing about
J1's torque during acceleration or under a lateral force.

The model uses URDF torque limits `(27,27,27,7,7,7) N·m`. The static panel's
default selected headroom fraction is `rho=0.7`, giving screening caps
`c_j=rho*tau_lim,j`, not validated continuous motor ratings. It reports

\[
u_j=\frac{|\tau_{hold,j}|}{c_j},\qquad
\text{headroom}_j=c_j-|\tau_{hold,j}|.
\]

The signed feasible mass interval is the intersection

\[
m\ge0,\qquad -c_j\le G_j+m a_j\le c_j
\quad\text{for every joint }j.
\]

An `a_j=0` axis either imposes no mass restriction or makes the pose infeasible
under its chosen cap. For nonzero `a_j`, solve both linear inequalities with
the appropriate sign and intersect all intervals. Added mass can counteract
gravity on one axis; if the unloaded pose is infeasible but only a positive
mass interval passes, it is labeled **counterweight-only**, not recommended
as a safe payload. Pose and COM offset can change which axis is limiting.
Six motor torque capacities must not simply be added to obtain payload or force.
The currently most loaded joint has the largest utilization, but another joint
can limit the feasible interval when payload mass is increased.

The interactive 0–4 kg slider explores model stress cases, not an approved
hardware payload range. Seeed's [B601-DM product page](https://www.seeedstudio.com/reBot-Arm-B601-DM-p-6740.html)
lists **1.5 kg, without gripper, with 70% reach recommended**. A larger
torque-only model interval does not supersede that qualified specification.

A conceptual static screening set is

\[
\mathcal C_{geom}=\{q:h_i(q)\ge h_0\ \forall i,\ \text{valid domain}\},
\quad
\mathcal C_{load}=\{q:|\tau_{hold,j}(q,m)|\le c_j\ \forall j\},
\quad
\mathcal C_{screen}=\mathcal C_{geom}\cap\mathcal C_{load}.
\]

The intersection is not a deployed torque filter. Changing the load-panel
controls does not add torque constraints to the running velocity CBF or change
the dynamic simulation's motor limits.

## 6. Distinguish capacity exceedance, controller failure and probability

| Reported quantity or event | Interpretation |
|---|---|
| `u_j > 1` | Static demand exceeds a selected deterministic model cap. |
| `u_j < 1` | Those particular static model inequalities pass. |
| QP feasible | A commanded velocity meets the modeled geometric constraints. |
| OSQP iteration limit | A numerical convergence event; not proof of infeasibility. |
| Progress stall | A logged policy trigger to replan; not proof all routes are blocked. |
| Actual contact or field violation | Failed simulation episode. |

Torque saturation can make an overloaded arm fail to track an otherwise
feasible velocity. Yet passing static caps does not validate thermal duty cycle,
gearing, grasp friction, payload inertia, structural strength, impact forces,
sensor error, or guaranteed braking. The point-mass screen also omits gripper
holding force and a changing payload geometry.

A failure probability requires calibrated distributions of uncertain loads,
parameters, strength and operating conditions, together with data. No such
stochastic model is fitted here. Utilization and headroom are deterministic
screening values, not probabilities of damage or success.

A human arm also balances joint moments, but motor torque cannot be converted
directly into muscle force. Unknown muscle moment arms, multiple agonists and
antagonists, co-contraction, posture, fatigue and individual physiology matter.
The shoulder/elbow grouping is a mechanical analogy, not a human strength,
fatigue or injury prediction.

## 7. Extend the model before enforcing actuator-aware dynamic safety

An external-force sign convention consistent with the static screen is

\[
M(q)\ddot q+C(q,\dot q)\dot q+G(q)=\tau+J^TF_{ext}.
\]

For `b(q,t)=h(p(q),t)`, torque first appears in its second derivative. One
relative-degree-two condition is

\[
\ddot b+(k_1+k_2)\dot b+k_1k_2 b\ge0.
\]

It requires field Hessians, Jacobian derivatives, dynamics and any field time
derivatives, as well as torque constraints and an admissible initial
position/velocity state. The current first-order CBF and static load screen do
not implement these requirements.

A mobile manipulator additionally needs base/torso/payload geometry, world
Jacobians, localization uncertainty, nonholonomic wheel constraints, coordinated
base–arm planning, braking and tipping/support constraints. Differential drive
cannot command an independent sideways velocity. Retain the hierarchy, but
validate a genuinely new whole-body model rather than only adding arm samples.


For a static environment, let \(b(q)=h(p(q))-h_0\). Then

\[
\dot b=\nabla_q b^T\dot q,\qquad
\ddot b=\dot q^T\nabla_q^2b\,\dot q+
\nabla_q b^TM^{-1}(\tau-C\dot q-G+J^TF_{ext}).
\]

This makes the proposed second-order constraint affine in commanded torque.
The exponential construction also requires the initial conditions
\(b(q_0)\ge0\) and \(\dot b(q_0,\dot q_0)+k_1b(q_0)\ge0\),
plus feasible torque constraints and the assumed regularity. For moving payloads,
include their inertia and gravity consistently in the dynamics; do not count
payload gravity both inside G and as a second external load.

For a human analogy, a hypothetical single muscle with a 0.03 m moment arm
would need 10/0.03 ≈ 333 N to provide 10 N·m of joint moment. That example
illustrates leverage only: actual muscle moment arms, recruitment and strength
are not measured by this robot model.


A load-conditioned positional reach set is R(m) = {p_TCP(q): q lies in the
joint-limit domain and passes both C_geom and C_load(m)}. It is a projection
of a configuration set. Two elbow arrangements at the same TCP position can
have different collision clearance and torque demand. Orientation, grasp,
payload geometry and a connecting path impose additional conditions. The
existing sampled reach map does not yet enforce these extra load conditions.
