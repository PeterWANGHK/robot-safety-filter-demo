# Paper reproduction and mobile-manipulator limitations

This experiment is a preliminary, independent implementation inspired by
Wilkinson et al., **“Full-Body Dynamic Safety for Robot Manipulators: 3D Poisson
Safety Functions for CBF-Based Safety Filters”**. The reference is the supplied
[arXiv v1 paper](https://arxiv.org/html/2604.21189v1); an
[IROS 2026 proceedings version](https://events.infovaya.com/uploads/documents/pdfviewer/97/77/232823-3920.pdf)
also exists. Section and equation numbers below refer to v1. This repository does
not reproduce the authors' hardware results or claim their continuous-time
safety theorem for the discretized simulation.

## High degrees of freedom versus higher relative degree

The paper's implemented controller is **first order**: Section II-B, equation (5),
assumes direct joint-velocity control, `qdot = v`. Its seven-joint FR3 is a
high-DOF robot; this does not make its controller a higher-order CBF (HOCBF).
Our six-joint B601 implementation also filters velocities. A genuine
relative-degree-two acceleration/torque formulation is described below as a
future extension and is **not implemented**.

The paper transfers kinematic safety to full dynamics conditional on accurate
low-level velocity tracking. Our saturated PD plant does not establish that
condition mathematically. Simulated tracking, clearance and intervention logs
are empirical evidence within the tested scene.

## Equations and implementation

| Stage | Paper/model relation | Code |
|---|---|---|
| Position reachability | Additional hierarchy stage; sampled forward kinematics, not paper algorithm | `planning.py: ReachabilityMap` |
| Collision-aware planning | Additional seeded joint-space RRT-Connect stage | `planning.py: RRTConnect`, `CollisionChecker` |
| Environment representation | Sections II-C/III-C, equations (9)/(16) | `field.py: PoissonSafetyField` |
| Surface coverage | Section III-A, equations (10)–(12) | `robot.py: _box_surface`, `RebotModel` |
| Buffered domain | Section III-B, equations (13)/(14) | `field.py`, `scenes.py` |
| CBF velocity filter | Section II-B, equations (7)/(8); Section III-D MC-CBF-QP | `controller.py: PoissonCBFFilter` |
| Low-level dynamics | Additional simulator adaptation | `robot.py: RebotModel.step`, `experiment.py` |

The field solves

\[
\Delta h(y)=-1\quad\text{in buffered free space},\qquad
h(y)=0\quad\text{on its boundary}.
\]

The negative forcing gives positive interior values. With this forcing, `h` has
units **m²** and is not Euclidean clearance. We measure clearance separately
using MuJoCo distances between complete conservative link boxes and obstacles.
All outer grid faces are also Dirichlet boundaries.

For surface samples `y_i = phi_i(q)`, the first-order constraint is

\[
\nabla h(y_i)J_i(q)v+\partial_t h(y_i,t)\ge-\alpha h(y_i,t).
\]

The QP minimizes half the squared change from nominal joint velocity. It adds
joint-speed bounds and joint-limit barriers. Our optional tracking disturbance
bound, field reserve, row normalization, gains, tolerances and post-solve
residual checks are implementation choices. There is no safety slack or
post-solve clipping of accepted velocities. Constraints provably redundant over
the full bounded velocity box are excluded from the QP to improve numerical
conditioning; all surface constraints are still checked after solving.
An explicitly logged SLSQP fallback solves the same convex QP if OSQP fails to
converge; its output must pass the same complete residual checks. This extra
solver and its potentially longer runtime are our choices, not the paper's.
The default experiment uses a
static field, hence `h_t = 0`; accepting a derivative argument in the filter API
does not implement dynamic mapping or estimate that derivative.

Invalid occupancy/interpolation queries are rejected independently of the
scalar field. Assigning zero inside obstacles and testing only `h >= 0` would
incorrectly declare their interiors safe. The implementation instead returns
negative invalid values and aborts filtering outside buffered free space.

## Choices that differ from the paper

The paper's Section IV-A uses a 100³ grid, checkerboard SOR with warm starts,
OSQP, and an RTX 5090. Our CPU implementation uses a seven-point finite
difference Laplacian and conjugate gradients with a discrete-sine-transform
preconditioner. Constant forcing `-1` is our choice: v1 does not specify its
numeric forcing, all gains, convergence tolerances or a temporal-derivative
estimator. The paper's reported rates are not Jetson benchmarks.

Our known, synthetic static box scene is not a camera-derived occupancy map.
The paper's dynamic demonstration reconstructs a moving UR10e's occupancy
from measured joint states; its broader perception pipeline is future work.
The current experiment neither updates obstacles online nor handles unknown
human trajectories.

The [official project website source](https://github.com/fullbodysafety/fullbodysafety.github.io/blob/main/index.html)
still shows a disabled “Code soon” button when checked on 26 November 2026.
The public project repository contains website assets, not the controller
implementation. This is independent code, not a port of released author code;
the article's license does not license an unreleased implementation.

## Geometry, planning and guarantee boundaries

The paper's Theorem 1 assumes an ε-cover of the continuous surface:
every surface point is within ε of a retained sample. Merely separating points
by ε is insufficient without checking coverage. Here each robot visual mesh is
enclosed by a body-local box, and face-grid samples have a calculable cover
radius. Obstacles are padded by the requested cover reserve plus grid
allowances; requiring all eight interpolation corners to be free adds
conservatism. A further numerical collar ensures the field approaches zero
before invalid cells, avoiding an artificial positive-to-invalid jump.
Narrow passages can disappear. Trilinear interpolation has gradient
jumps at voxel faces, so it is not the smooth continuous field used by the
theorem.

The gripper is fixed open. There is no grasped payload, variable finger motion,
base motion or self-collision enforcement. The visual floor is excluded from
environment collision checking; the explicit table is included. These omissions
must accompany any “whole-arm” result: coverage concerns the declared proxy
geometry and modeled obstacles, not every physical contact possibility.

The reachability cache stores sampled TCP positions and joint seeds. Selection
establishes proximity to one stored position, not orientation, dexterity,
payload capacity, IK completeness or a connecting path. The task includes
disclosed known start/goal seeds. A missing cache entry does not prove
unreachability. RRT edges and shortcuts use finite joint interpolation; they
reduce tunneling but are not continuous-collision certificates.

A shared field does not guarantee compatible joint-space constraints. Moving
obstacles, joint bounds, insufficient mobility and actuator limits can make
the QP infeasible. The paper's Figure 3 itself reports infeasibility with large
buffers. Fixed-base samples have zero Jacobian and cannot escape an obstacle
approaching the mount. Reactive filters can also stall without finding another
homotopy route, which motivates the separate planner. Solver failure or a
tracking-domain violation is a failed episode, not successful safe manipulation;
a zero-velocity request is not a proof of safe inertial braking.

## What a true relative-degree-two extension requires

For `b(q,t) = h(phi(q),t)`, define `psi_1 = bdot + k1 b`. With acceleration
input `a = qddot`, a standard exponential HOCBF condition is

\[
\ddot b+(k_1+k_2)\dot b+k_1k_2b\ge0,
\]

\[
\ddot b=\nabla hJa+(J\dot q)^T\nabla_y^2h(J\dot q)
+\nabla h\dot J\dot q+2(\partial_t\nabla h)^TJ\dot q+\partial_{tt}h.
\]

Initially both `b >= 0` and `psi_1 >= 0` are required. For torque input,
substitute `a = M^-1(B tau - C qdot - G)` and impose torque limits. This needs
reliable dynamics, higher spatial/time derivatives and actuator-aware feasible
sets. Trilinear Hessians and integrating acceleration into a PD target do not
establish those requirements. Present torque telemetry is applied simulated PD
effort, not contact force regulation or a torque-level CBF.

## Extending to a mobile manipulator

For differential drive, use `z = (x_b, y_b, theta_b, q)` and
`u = (v_forward, omega, qdot)`:

\[
\dot z=S(z)u,\qquad
S(z)=\begin{bmatrix}\cos\theta_b&0&0\\
\sin\theta_b&0&0\\0&1&0\\0&0&I\end{bmatrix}.
\]

World-frame body-point Jacobians must be composed with `S(z)`. Independent
Cartesian base velocities would incorrectly model an ordinary wheeled base as
holonomic. `controller.py` provides this input-map interface only; it is not a
mobile-base simulation.

Extension requires base/torso/wheel and attached-payload samples, coordinated
base–arm planning, base–arm self-collision constraints, and a world-frame field
with localization and mapping uncertainty. Wheel slip, occlusion and map
latency need explicit error margins or models. Braking distance, acceleration
and torque saturation restrict recoverable states; a CBF cannot invent lateral
wheel motion or instant stopping. Payload centre of mass, tipping/support
polygon limits and contact forces require additional dynamics and constraints.
Retain the hierarchy while assessing these effects separately from fixed-base
geometric collision avoidance.
