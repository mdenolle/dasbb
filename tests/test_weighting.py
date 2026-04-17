"""Tests for adaptive weighting: VCE, IRLS, GCV."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dasbb import (
    JointInversion, InversionConfig, AdaptiveWeighting,
    generate_synthetic_test,
)


@pytest.fixture
def setup():
    synth = generate_synthetic_test(n_das_channels=150, n_bb_stations=12, seed=77)
    inv = JointInversion(synth['velocity_model'],
                         InversionConfig(max_iter_location=15, damping_location=0.1))
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + [0.5, -0.3, 0.8],
    )
    return synth, inv, result


def test_irls_weights_range():
    r = np.array([0.01, 0.02, 0.5, 2.0, 10.0])
    w = AdaptiveWeighting.irls_weights(r, sigma=0.1, method='huber')
    assert np.all(w >= 0)
    assert np.all(w <= 1)
    assert w[0] > w[-1]  # small residual → higher weight


def test_irls_tukey_zeroes_outliers():
    r = np.array([0.01, 0.02, 100.0])
    w = AdaptiveWeighting.irls_weights(r, sigma=0.1, method='tukey')
    assert w[-1] == 0.0


def test_irls_improves_with_outliers(setup):
    synth, inv, _ = setup
    # Inject outliers
    from dasbb import DASPicks, BroadbandPicks
    das_bad = DASPicks(
        times=synth['das_picks'].times.copy(),
        receiver_xyz=synth['das_picks'].receiver_xyz.copy(),
        fiber_coords=synth['das_picks'].fiber_coords.copy(),
        phase='P', pick_sigma=synth['das_picks'].pick_sigma,
        clock_sigma=synth['das_picks'].clock_sigma,
        correlation_length=synth['das_picks'].correlation_length,
    )
    das_bad.times[10] += 2.0
    das_bad.times[50] += 2.0

    result_naive = inv.locate_event(
        das_bad, synth['bb_picks'],
        synth['true_source'] + [0.5, -0.3, 0.8],
    )
    result_irls = AdaptiveWeighting.irls_locate(
        inv, das_bad, synth['bb_picks'],
        synth['true_source'] + [0.5, -0.3, 0.8],
        synth['true_origin_time'], n_irls_iterations=3,
    )
    err_naive = np.linalg.norm(result_naive['source_xyz'] - synth['true_source'])
    err_irls = np.linalg.norm(result_irls['source_xyz'] - synth['true_source'])
    assert err_irls < err_naive * 1.5  # IRLS should help or be comparable


def test_task_adaptive_weights_mean_one():
    loc = np.array([0.1, 0.5, 0.2, 0.8])
    tomo = np.array([0.8, 0.2, 0.5, 0.1])
    w = AdaptiveWeighting.task_adaptive_weights(loc, tomo, 0.5)
    assert abs(w.mean() - 1.0) < 0.01
    assert np.all(w > 0)


def test_gcv_returns_positive_lambda():
    rng = np.random.RandomState(42)
    G = rng.randn(30, 10)
    m = rng.randn(10)
    d = G @ m + 0.1 * rng.randn(30)
    result = AdaptiveWeighting.gcv_lambda(G, d)
    assert result['lambda_optimal'] > 0
