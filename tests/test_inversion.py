"""Tests for joint location and tomography inversion."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dasbb import (
    JointInversion, InversionConfig, generate_synthetic_test,
)


@pytest.fixture
def synth():
    return generate_synthetic_test(n_das_channels=200, n_bb_stations=15, seed=42)


@pytest.fixture
def inv(synth):
    config = InversionConfig(max_iter_location=20, damping_location=0.1,
                             compute_posterior=True)
    return JointInversion(synth['velocity_model'], config)


def test_joint_location_error(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([1.0, -0.5, 1.5]),
        synth['true_origin_time'] - 0.2,
    )
    err = np.linalg.norm(result['source_xyz'] - synth['true_source'])
    assert err < 3.0, f"Location error {err:.3f} km too large"


def test_origin_time_error(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([1.0, -0.5, 1.5]),
        synth['true_origin_time'] - 0.2,
    )
    tau_err = abs(result['origin_time'] - synth['true_origin_time'])
    assert tau_err < 1.0, f"Origin time error {tau_err*1000:.0f} ms too large"


def test_posterior_exists(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    assert result['posterior_cov'] is not None


def test_posterior_positive_definite(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    if result['posterior_cov'] is not None:
        eigvals = np.linalg.eigvalsh(result['posterior_cov'])
        assert np.all(eigvals > 0)


def test_ellipsoid_axes_positive(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    ell = result.get('location_ellipsoid')
    if ell is not None:
        assert np.all(ell['semi_axes_km'] > 0)
        assert np.all(ell['semi_axes_km'] < 50)


def test_fisher_psd(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    assert np.all(np.linalg.eigvalsh(result['F_das']) >= -1e-10)
    assert np.all(np.linalg.eigvalsh(result['F_bb']) >= -1e-10)


def test_information_decomposition_sums_to_one(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    decomp = inv.information_decomposition(result['F_das'], result['F_bb'])
    assert len(decomp) == 4
    for name, info in decomp.items():
        total = info['das_fraction'] + info['bb_fraction']
        assert abs(total - 1.0) < 0.01, f"{name}: fractions sum to {total}"


def test_das_only_location(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], None,
        synth['true_source'] + np.array([1.0, -0.5, 1.5]),
        synth['true_origin_time'],
    )
    err = np.linalg.norm(result['source_xyz'] - synth['true_source'])
    assert err < 15.0, f"DAS-only error {err:.3f} km (poor depth expected)"


def test_bb_only_location(synth, inv):
    result = inv.locate_event(
        None, synth['bb_picks'],
        synth['true_source'] + np.array([1.0, -0.5, 1.5]),
        synth['true_origin_time'],
    )
    err = np.linalg.norm(result['source_xyz'] - synth['true_source'])
    assert err < 5.0, f"BB-only error {err:.3f} km"


def test_chi2_keys(synth, inv):
    result = inv.locate_event(
        synth['das_picks'], synth['bb_picks'],
        synth['true_source'] + np.array([0.5, -0.3, 0.8]),
    )
    assert 'das' in result['chi2']
    assert 'bb' in result['chi2']
    assert result['chi2']['das'] > 0
    assert result['chi2']['bb'] > 0
