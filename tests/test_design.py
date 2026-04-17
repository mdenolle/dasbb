"""Tests for optimal design and sensor placement."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dasbb import (
    VelocityModel, OptimalDesign, generate_synthetic_test,
    JointInversion, InversionConfig,
)


@pytest.fixture
def synth():
    return generate_synthetic_test(n_das_channels=100, n_bb_stations=10, seed=7)


@pytest.fixture
def od(synth):
    return OptimalDesign(synth['velocity_model'])


def test_station_fisher_psd(od, synth):
    F = od.compute_station_fisher(
        synth['true_source'], synth['bb_xyz'][:1], 'P', 0.05
    )
    assert F.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(F) >= -1e-10)


def test_d_optimal_finite(od, synth):
    inv = JointInversion(synth['velocity_model'], InversionConfig(max_iter_location=10))
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + [0.5, -0.3, 0.8],
    )
    F = result['F_das'] + result['F_bb']
    assert np.isfinite(od.d_optimal_criterion(F))


def test_greedy_monotonic(od, synth):
    inv = JointInversion(synth['velocity_model'], InversionConfig(max_iter_location=10))
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + [0.5, -0.3, 0.8],
    )
    F = result['F_das'] + result['F_bb']
    rng = np.random.RandomState(42)
    candidates = np.column_stack([1 + 27 * rng.rand(30), 1 + 27 * rng.rand(30), np.zeros(30)])

    greedy = od.greedy_optimal_placement(
        target_sources=np.array([synth['true_source']]),
        candidate_positions=candidates, existing_F=F,
        n_to_add=3, sigma=0.05, criterion='D',
    )
    hist = greedy['criterion_history']
    assert all(hist[i] <= hist[i+1] + 1e-10 for i in range(len(hist)-1))


def test_azimuthal_gap(od, synth):
    gap = od.azimuthal_gap_analysis(synth['true_source'], synth['bb_xyz'])
    assert 0 < gap['max_gap_deg'] <= 360
    assert 1 <= gap['n_quadrants'] <= 4


def test_geometry_weights(od, synth):
    w = od.compute_geometry_weights(
        synth['true_source'], synth['bb_xyz'], 'P',
        synth['bb_picks'].uncertainties,
    )
    assert len(w) == len(synth['bb_xyz'])
    assert np.all(w >= 0)
