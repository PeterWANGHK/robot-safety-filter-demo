import numpy as np
import pytest
from cbf_demo.controller import PoissonCBFFilter, differential_drive_input_map


class PlaneField:
    def sample(self, p):
        return p[:, 0], np.tile([1., 0., 0.], (len(p), 1))

    def is_valid(self, p):
        return p[:, 0] > 0


def test_wall_constraint_and_minimal_intervention():
    f = PoissonCBFFilter(PlaneField(), [-2.], [2.], alpha=2,
                         h_reserve=0, speed=1)
    j = np.array([[[1.], [0.], [0.]]])
    r = f.filter([0.1], [-1.], [[0.1, 0., 0.]], j)
    assert r.success
    assert abs(r.velocity[0] + .2) < 1e-5
    away = f.filter([.1], [.3], [[.1, 0., 0.]], j)
    assert away.success and abs(away.velocity[0] - .3) < 1e-5


def test_dynamic_derivative_and_impossible_static_body():
    f = PoissonCBFFilter(PlaneField(), [-2.], [2.], h_reserve=0, speed=1)
    r = f.filter([0.], [0.], [[.1, 0., 0.]], np.zeros((1, 3, 1)),
                 field_time_derivative=[-.5])
    assert not r.success and np.array_equal(r.velocity, [0.])


def test_occupied_and_nan_inputs_rejected():
    f = PoissonCBFFilter(PlaneField(), [-2.], [2.])
    j = np.zeros((1, 3, 1))
    assert not f.filter([0], [0], [[-.1, 0, 0]], j).success
    assert not f.filter([0], [np.nan], [[.1, 0, 0]], j).success


def test_joint_limits_are_hard_and_not_post_clipped():
    f = PoissonCBFFilter(PlaneField(), [-1.], [1.], speed=1, joint_margin=.05)
    r = f.filter([.94], [1.], [[.1, 0, 0]], np.zeros((1, 3, 1)))
    assert r.success and r.velocity[0] <= .020001


def test_differential_drive_has_no_lateral_actuation():
    s = differential_drive_input_map(np.pi/2)
    z = s @ np.r_[.4, .2, np.zeros(6)]
    assert np.allclose(z[:3], [0, .4, .2])
    assert s.shape == (9, 8)


def test_nonfinite_configuration_rejected_before_solver():
    for option in ['alpha','speed','joint_margin','h_reserve','tracking_speed_bound']:
        with pytest.raises(ValueError):
            PoissonCBFFilter(PlaneField(),[-1.],[1.],**{option:float('nan')})


def test_exact_box_redundancy_preserves_full_constraint_solution():
    f = PoissonCBFFilter(PlaneField(),[-2.],[2.],speed=1,h_reserve=0)
    j = np.array([[[1.],[0.],[0.]],[[1.],[0.],[0.]],[[0.],[0.],[0.]]])
    r = f.filter([0.],[-1.],[[.1,0,0],[.2,0,0],[4,0,0]],j)
    assert r.success and abs(r.velocity[0]+.2)<1e-5
    assert r.total_constraints==3 and r.qp_constraints==2
    assert r.minimum_residual>=-2e-6


def test_iteration_limit_fallback_solves_same_convex_qp(monkeypatch):
    from types import SimpleNamespace
    import cbf_demo.controller as module
    class FailedOSQP:
        def setup(self,**kwargs): pass
        def solve(self,**kwargs):
            return SimpleNamespace(x=np.array([0.]),info=SimpleNamespace(
                status_val=7,status='maximum iterations reached'))
    monkeypatch.setattr(module.osqp,'OSQP',FailedOSQP)
    f=PoissonCBFFilter(PlaneField(),[-2.],[2.],speed=1,h_reserve=0)
    r=f.filter([0.],[-1.],[[.1,0,0]],np.array([[[1.],[0.],[0.]]]))
    assert r.success and r.fallback_used and abs(r.velocity[0]+.2)<1e-6
    assert r.minimum_residual >= -2e-6
