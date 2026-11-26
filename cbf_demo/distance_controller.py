"""One-step robot-distance CBF comparison; independent of the Poisson filter.

For every static obstacle/proxy-box pair k, h_k=d_k-margin (metres) and
grad_q(d_k) v >= -alpha*h_k. The QP minimizes 1/2||v-v_nom||^2 under those
rows and the same velocity/joint-margin bounds used by PoissonCBFFilter.

Only constraints proven redundant over the entire admissible velocity box
are omitted from the numerical QP; EVERY original pair row is post-checked.
There is no safety slack and no post-solve clipping. These local, piecewise
distance constraints do not establish torque-level, tracking, sampled-data,
moving-obstacle or mobile-base safety. This module initially supports a
one-step interactive comparison, not an evaluated closed-loop baseline.

FilterResult.minimum_h is in metres here; Poisson's value is in m^2. Their
scalar values and residuals must not be compared as equivalent clearances.
"""
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, minimize

from .controller import FilterResult
from .robot_field import pair_clearance_constraints


class RobotDistanceCBFFilter:
    barrier_units = 'm'
    barrier_definition = 'physical_proxy_pair_clearance_minus_margin'
    comparison_only = True
    formal_safety_certificate = False

    def __init__(self, robot, *, margin=.05, alpha=2., speed=.45, joint_margin=.015):
        self.robot = robot
        self.lower = np.asarray(robot.lower, dtype=float).copy()
        self.upper = np.asarray(robot.upper, dtype=float).copy()
        self.margin, self.alpha = float(margin), float(alpha)
        self.joint_margin = float(joint_margin)
        self.speed = np.broadcast_to(speed, self.lower.shape).astype(float).copy()
        if (self.lower.ndim != 1 or not len(self.lower) or self.upper.shape != self.lower.shape
                or not all(np.all(np.isfinite(value)) for value in
                           (self.lower, self.upper, self.speed, self.margin, self.alpha, self.joint_margin))
                or self.margin < 0 or self.alpha <= 0 or np.any(self.speed <= 0)
                or self.joint_margin < 0
                or np.any(self.lower+self.joint_margin >= self.upper-self.joint_margin)):
            raise ValueError('invalid distance-filter parameters')
        self.last_geometry = None
        self.last_diagnostics = {}

    def filter(self, q, nominal):
        begin = perf_counter()
        q, nominal = np.asarray(q, dtype=float), np.asarray(nominal, dtype=float)
        n = len(self.lower)
        self.last_geometry = None
        self.last_diagnostics = {'barrier_units': 'm', 'one_step_comparison_only': True,
                                 'formal_safety_certificate': False}
        if q.shape != (n,) or nominal.shape != (n,):
            raise ValueError('q and nominal must match the robot joint dimension')
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(nominal)):
            return self._failure(n, 'nonfinite_input', begin)
        geometry = pair_clearance_constraints(self.robot, q)
        self.last_geometry = geometry
        distances = np.asarray(geometry['distances_m'], dtype=float)
        a = np.asarray(geometry['gradients_q'], dtype=float)
        count = len(distances)
        if (not geometry['valid'] or a.shape != (count, n)
                or not np.all(np.isfinite(distances)) or not np.all(np.isfinite(a))):
            minimum = float(np.min(distances)-self.margin) if count else float('inf')
            return self._failure(n, 'invalid_or_contact_pair_geometry', begin, minimum, count)
        h = distances-self.margin
        minimum_h = float(h.min()) if count else float('inf')
        self.last_diagnostics.update(total_pair_constraints=count,
                                     initial_margin_feasible=bool(np.all(h >= 0)),
                                     minimum_clearance_m=float(distances.min()) if count else None,
                                     margin_m=self.margin)
        b = -self.alpha*h
        scale = np.maximum(np.linalg.norm(a, axis=1), 1e-3)
        an, bn = a/scale[:, None], b/scale
        lo = np.maximum(-self.speed, -self.alpha*(q-self.lower-self.joint_margin))
        hi = np.minimum(self.speed, self.alpha*(self.upper-self.joint_margin-q))
        if np.any(lo > hi):
            return self._failure(n, 'infeasible_joint_limits', begin, minimum_h, count)
        box_minimum = np.where(an >= 0, an*lo, an*hi).sum(axis=1)
        keep = box_minimum < bn+1e-10
        qp_count = int(np.count_nonzero(keep))
        matrix = sparse.csc_matrix(np.vstack([an[keep], np.eye(n)]))
        lower, upper = np.r_[bn[keep], lo], np.r_[np.full(qp_count, np.inf), hi]
        solver = osqp.OSQP()
        solver.setup(P=sparse.eye(n, format='csc'), q=-nominal, A=matrix,
                     l=lower, u=upper, verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                     max_iter=6000, polishing=False, check_termination=10, rho=1.)
        result = solver.solve(raise_error=False)
        fallback_used = result.info.status_val not in (1, 2) or result.x is None
        solve_status = result.info.status
        if fallback_used:
            x0 = np.clip(np.zeros(n), lo, hi)
            if result.x is not None and np.all(np.isfinite(result.x)):
                x0 = np.clip(result.x, lo, hi)
            constraints = [LinearConstraint(an[keep], bn[keep], np.inf)] if np.any(keep) else []
            recovered = minimize(lambda x: .5*np.sum((x-nominal)**2), x0,
                                 jac=lambda x: x-nominal, method='SLSQP',
                                 bounds=Bounds(lo, hi), constraints=constraints,
                                 options=dict(ftol=1e-10, maxiter=150, disp=False))
            if not recovered.success:
                return self._failure(n, 'qp_'+result.info.status.replace(' ', '_')+'_fallback_failed',
                                     begin, minimum_h, count, qp_count, True)
            velocity = np.asarray(recovered.x)
            solve_status = 'solved_slsqp_fallback'
        else:
            velocity = np.asarray(result.x)
        residual = an@velocity-bn
        minimum_residual = min(float(residual.min()) if count else float('inf'),
                               float((velocity-lo).min()), float((hi-velocity).min()))
        if not np.all(np.isfinite(velocity)) or minimum_residual < -2e-6:
            return self._failure(n, 'qp_residual_rejected', begin, minimum_h,
                                 count, qp_count, fallback_used)
        self.last_diagnostics.update(qp_pair_constraints=qp_count,
                                     all_pair_rows_postchecked=True,
                                     normalized_residual_units='rad/s',
                                     fallback_used=bool(fallback_used))
        return FilterResult(velocity, True, solve_status, minimum_h, minimum_residual,
                            float(np.linalg.norm(velocity-nominal)), (perf_counter()-begin)*1000,
                            int(np.count_nonzero(residual < 1e-4)), count, qp_count, fallback_used)

    @staticmethod
    def _failure(n, status, begin, minimum_h=float('nan'), count=0, qp_count=0, fallback=False):
        # A zero request is not a verified collision-free braking trajectory.
        return FilterResult(np.zeros(n), False, status, minimum_h, float('nan'),
                            0., (perf_counter()-begin)*1000, 0, count, qp_count, fallback)
