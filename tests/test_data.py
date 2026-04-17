"""Tests for data structures and covariance builders."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dasbb import DASPicks, BroadbandPicks, VelocityModel


@pytest.fixture
def das_picks():
    n = 500
    fiber_xi = np.linspace(0, 10.0, n)
    return DASPicks(
        times=np.zeros(n), receiver_xyz=np.column_stack([fiber_xi, np.zeros(n), np.zeros(n)]),
        fiber_coords=fiber_xi, phase='P',
        pick_sigma=0.003, clock_sigma=0.001, correlation_length=0.05,
    )


@pytest.fixture
def bb_picks():
    n = 20
    rng = np.random.RandomState(42)
    return BroadbandPicks(
        times=np.zeros(n),
        receiver_xyz=rng.rand(n, 3) * 30,
        phases=['P'] * n,
        uncertainties=0.02 + 0.06 * rng.rand(n),
    )


def test_das_covariance_symmetric(das_picks):
    C = das_picks.build_covariance()
    assert np.allclose(C, C.T, atol=1e-12)


def test_das_covariance_positive_definite(das_picks):
    C = das_picks.build_covariance()
    assert np.all(np.linalg.eigvalsh(C) > 0)


def test_das_neff_less_than_n(das_picks):
    n_eff = das_picks.effective_n()
    assert n_eff < 0.5 * das_picks.n_picks
    assert n_eff > 5


def test_bb_covariance_diagonal(bb_picks):
    C = bb_picks.build_covariance()
    assert np.allclose(C, np.diag(np.diag(C)))


def test_bb_inverse(bb_picks):
    C = bb_picks.build_covariance()
    C_inv = bb_picks.build_covariance_inverse()
    assert np.allclose(C @ C_inv, np.eye(bb_picks.n_picks), atol=1e-10)


def test_velocity_model_slowness():
    vp = 5.0 * np.ones((5, 5, 5))
    vs = vp / 1.73
    vm = VelocityModel(vp=vp, vs=vs, origin=np.zeros(3), spacing=np.ones(3))
    assert np.allclose(vm.slowness('P'), 1.0 / 5.0)
    assert np.allclose(vm.slowness('S'), 1.73 / 5.0)


def test_velocity_model_grid_coordinates():
    vm = VelocityModel(
        vp=np.ones((10, 8, 6)), vs=np.ones((10, 8, 6)),
        origin=np.array([1., 2., 3.]), spacing=np.array([0.5, 1.0, 0.25])
    )
    x, y, z = vm.grid_coordinates()
    assert len(x) == 10
    assert x[0] == 1.0
    assert y[-1] == pytest.approx(2.0 + 7 * 1.0)
    assert z[-1] == pytest.approx(3.0 + 5 * 0.25)
