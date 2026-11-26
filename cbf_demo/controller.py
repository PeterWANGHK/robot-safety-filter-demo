"""Hard multi-point Poisson CBF constraints, with bounded joint velocities.

Paper equations (5), (7), (8), (16): qdot=v and grad(h) J v + h_t >= -alpha h.
Numerical choices here (normalization, reserve, gains, tolerances) are ours.
No safety slack, post-solve velocity clipping, or unchecked solver output.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass
class FilterResult:
    velocity: np.ndarray
    success: bool
    status: str
    minimum_h: float
    minimum_residual: float
    correction_norm: float
    solve_ms: float
    active_constraints: int
    total_constraints: int = 0
    qp_constraints: int = 0
    fallback_used: bool = False


class PoissonCBFFilter:
    def __init__(self, field, lower, upper, *, alpha=2.0, speed=0.45,
                 joint_margin=0.015, h_reserve=2e-5, tracking_speed_bound=0.0):
        self.field = field
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.alpha = float(alpha)
        self.speed = np.broadcast_to(speed, self.lower.shape).astype(float).copy()
        self.joint_margin = float(joint_margin)
        self.h_reserve = float(h_reserve)
        self.tracking_speed_bound = float(tracking_speed_bound)
        if (self.lower.ndim != 1 or self.upper.shape != self.lower.shape
                or not all(np.all(np.isfinite(a)) for a in
                           [self.lower,self.upper,self.speed,self.alpha,
                            self.joint_margin,self.h_reserve,self.tracking_speed_bound])
                or np.any(self.upper <= self.lower) or self.alpha <= 0
                or np.any(self.speed <= 0) or self.h_reserve < 0
                or self.joint_margin < 0 or self.tracking_speed_bound < 0
                or np.any(self.lower+self.joint_margin >= self.upper-self.joint_margin)):
            raise ValueError('invalid filter parameters')

    def filter(self, q, nominal, points, jacobians, *, field_time_derivative=None):
        begin = perf_counter()
        q = np.asarray(q, dtype=float)
        nominal = np.asarray(nominal, dtype=float)
        points = np.asarray(points, dtype=float)
        jacobians = np.asarray(jacobians, dtype=float)
        n = len(self.lower)
        if (q.shape != (n,) or nominal.shape != (n,) or points.ndim != 2
                or points.shape[1] != 3 or jacobians.shape != (len(points), 3, n)
                or not len(points)):
            raise ValueError('inconsistent q/velocity/point/Jacobian shapes')
        if not all(np.all(np.isfinite(a)) for a in [q, nominal, points, jacobians]):
            return self._failure(n, 'nonfinite_input', begin)
        values, gradients = self.field.sample(points)
        values, gradients = np.asarray(values), np.asarray(gradients)
        if (not np.all(self.field.is_valid(points)) or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(gradients)) or np.any(values <= 0)):
            return self._failure(n, 'outside_buffered_free_space', begin,
                                 float(np.min(values)))
        ht = (np.zeros(len(points)) if field_time_derivative is None
              else np.asarray(field_time_derivative, dtype=float))
        if ht.shape != values.shape or not np.all(np.isfinite(ht)):
            return self._failure(n, 'invalid_field_time_derivative', begin)
        # A bound on Cartesian sample-point tracking error, when supplied,
        # subtracts the worst possible disturbance grad(h) e from hdot.
        a = np.einsum('ni,nij->nj', gradients, jacobians)
        b = (-self.alpha * (values - self.h_reserve) - ht
             + self.tracking_speed_bound * np.linalg.norm(gradients, axis=1))
        scale = np.maximum(np.linalg.norm(a, axis=1), 1e-3)
        an, bn = a / scale[:, None], b / scale
        lo = np.maximum(-self.speed,
                        -self.alpha * (q - self.lower - self.joint_margin))
        hi = np.minimum(self.speed,
                        self.alpha * (self.upper - self.joint_margin - q))
        if np.any(lo > hi):
            return self._failure(n, 'infeasible_joint_limits', begin, float(values.min()))
        # Remove only constraints proven redundant over the ENTIRE admissible
        # velocity box. This is exact, not geometric sample downsampling.
        # Dense inactive/zero rows otherwise cause OSQP convergence problems.
        # Every original surface constraint is still verified after the solve.
        box_minimum = np.where(an >= 0, an*lo, an*hi).sum(axis=1)
        keep = box_minimum < bn + 1e-10
        matrix = sparse.csc_matrix(np.vstack([an[keep], np.eye(n)]))
        l = np.r_[bn[keep], lo]
        u = np.r_[np.full(np.count_nonzero(keep), np.inf), hi]
        solver = osqp.OSQP()
        solver.setup(P=sparse.eye(n, format='csc'), q=-nominal, A=matrix,
                     l=l, u=u, verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                     max_iter=6000, polishing=False, check_termination=10, rho=1.0)
        result = solver.solve(raise_error=False)
        fallback_used = result.info.status_val not in (1, 2) or result.x is None
        solve_status = result.info.status
        if fallback_used:
            # Numerical fallback for this small convex QP; never interpret
            # OSQP iteration exhaustion as proof of geometric infeasibility.
            # The objective, constraints and ALL-row acceptance checks remain
            # identical. This explicit fallback is an implementation choice.
            x0 = np.clip(np.zeros(n), lo, hi)
            if result.x is not None and np.all(np.isfinite(result.x)):
                x0 = np.clip(result.x, lo, hi)
            constraints = ([LinearConstraint(an[keep],bn[keep],np.inf)]
                           if np.any(keep) else [])
            recovered = minimize(lambda x: .5*np.sum((x-nominal)**2),x0,
                                 jac=lambda x: x-nominal,method='SLSQP',
                                 bounds=Bounds(lo,hi),constraints=constraints,
                                 options=dict(ftol=1e-10,maxiter=150,disp=False))
            if not recovered.success:
                return self._failure(n, 'qp_' + result.info.status.replace(' ', '_')
                                     + '_fallback_failed',begin,float(values.min()))
            v = np.asarray(recovered.x)
            solve_status = 'solved_slsqp_fallback'
        else:
            v = np.asarray(result.x)
        residual = an @ v - bn
        # Independent post-solve check, including joint speed/limit bounds.
        min_residual = min(float(residual.min()), float((v-lo).min()),
                           float((hi-v).min()))
        if not np.all(np.isfinite(v)) or min_residual < -2e-6:
            return self._failure(n, 'qp_residual_rejected', begin, float(values.min()))
        return FilterResult(v, True, solve_status, float(values.min()),
                            min_residual, float(np.linalg.norm(v - nominal)),
                            (perf_counter() - begin) * 1000,
                            int(np.count_nonzero(residual < 1e-4)),
                            len(points), int(np.count_nonzero(keep)),fallback_used)

    @staticmethod
    def _failure(n, status, begin, h=float('nan')):
        # Zero requests braking; it is not proof of collision-free braking
        # for an inertial robot or for an approaching obstacle.
        return FilterResult(np.zeros(n), False, status, h, float('nan'),
                            0.0, (perf_counter()-begin)*1000, 0)


def differential_drive_input_map(heading, arm_dof=6):
    """For z=(x,y,theta,q), zdot=S(z)(v_forward,omega,qdot).

    This tested mapping is an extension interface, not a mobile-base demo.
    Compose each world-point Jacobian with this S before constructing a CBF.
    """
    if not np.isfinite(heading) or arm_dof < 1:
        raise ValueError('invalid heading or arm DOF')
    s = np.zeros((3 + arm_dof, 2 + arm_dof))
    s[0, 0], s[1, 0] = np.cos(heading), np.sin(heading)
    s[2, 1] = 1
    s[3:, 2:] = np.eye(arm_dof)
    return s
