"""Tests for the eikonal forward engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dasbb import VelocityModel, EikonalSolver


@pytest.fixture
def simple_model():
    vp = 5.5 * np.ones((20, 20, 10))
    vs = vp / 1.73
    return VelocityModel(vp=vp, vs=vs, origin=np.array([0.,0.,0.]),
                         spacing=np.array([1.,1.,1.]))


def test_eikonal_travel_time_positive(simple_model):
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[2., 2., 0.], [18., 18., 0.]])
    tt = solver.travel_times_at_receivers(src, rx, 'P')
    assert np.all(tt > 0)


def test_travel_time_correlates_with_distance(simple_model):
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[5., 5., 0.], [10., 10., 0.5], [15., 15., 0.]])
    tt = solver.travel_times_at_receivers(src, rx, 'P')
    dist = np.linalg.norm(rx - src, axis=1)
    assert np.corrcoef(dist, tt)[0, 1] > 0.9


def test_apparent_velocity_reasonable(simple_model):
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[2., 2., 0.]])
    tt = solver.travel_times_at_receivers(src, rx, 'P')
    dist = np.linalg.norm(rx[0] - src)
    v_app = dist / tt[0]
    assert 3.0 < v_app < 8.0


def test_frechet_source_nonzero(simple_model):
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[5., 5., 0.], [15., 5., 0.]])
    G = solver.frechet_source(src, rx, 'P')
    assert G.shape == (2, 3)
    assert np.max(np.abs(G)) > 0.01


def test_frechet_source_sign_convention(simple_model):
    """Moving source toward receiver should decrease travel time (negative derivative)."""
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[5., 10., 0.]])  # receiver at x=5
    G = solver.frechet_source(src, rx, 'P')
    # ∂T/∂x_s should be positive (moving source in +x moves it away from rx at x=5)
    assert G[0, 0] > 0


def test_s_wave_slower_than_p(simple_model):
    solver = EikonalSolver(simple_model)
    src = np.array([10., 10., 5.])
    rx = np.array([[5., 5., 0.]])
    tt_p = solver.travel_times_at_receivers(src, rx, 'P')
    tt_s = solver.travel_times_at_receivers(src, rx, 'S')
    assert tt_s[0] > tt_p[0]
