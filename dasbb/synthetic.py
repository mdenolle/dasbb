"""
Synthetic test scenario generators with known ground truth.

Realistic configurations for Alaska DAS and ocean island volcanic
environments, with correlated DAS noise and heteroscedastic BB noise.
"""

import numpy as np
from typing import Optional, Dict, Any

from .data import DASPicks, BroadbandPicks, VelocityModel
from .forward import EikonalSolver


def generate_synthetic_test(
    n_das_channels: int = 500,
    n_bb_stations: int = 30,
    fiber_length_km: float = 10.0,
    network_aperture_km: float = 50.0,
    true_source: Optional[np.ndarray] = None,
    noise_level: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generic synthetic test with a crustal velocity model.

    Returns dict with das_picks, bb_picks, velocity_model,
    true_source, true_origin_time, das_xyz, bb_xyz.
    """
    rng = np.random.RandomState(seed)
    if true_source is None:
        true_source = np.array([15.0, 15.0, 8.0])

    nx, ny, nz = 30, 30, 15
    vp = np.zeros((nx, ny, nz))
    for iz in range(nz):
        vp[:, :, iz] = 5.0 + 0.1 * iz
    vp[12:18, 12:18, 5:10] *= 0.92
    vs = vp / 1.73

    vm = VelocityModel(
        vp=vp, vs=vs,
        origin=np.array([0.0, 0.0, 0.0]),
        spacing=np.array([1.0, 1.0, 1.0])
    )

    fiber_xi = np.linspace(0, fiber_length_km, n_das_channels)
    das_xyz = np.column_stack([
        3.0 + fiber_xi * np.cos(np.radians(20)),
        5.0 + fiber_xi * np.sin(np.radians(20)),
        np.zeros(n_das_channels)
    ])
    das_xyz[:, 0] = np.clip(das_xyz[:, 0], 0.5, 28.5)
    das_xyz[:, 1] = np.clip(das_xyz[:, 1], 0.5, 28.5)

    angles = np.linspace(0, 2 * np.pi, n_bb_stations, endpoint=False)
    radii = (network_aperture_km / 5) + (network_aperture_km * 0.4) * rng.rand(n_bb_stations)
    bb_xyz = np.column_stack([
        15 + radii * np.cos(angles),
        15 + radii * np.sin(angles),
        np.zeros(n_bb_stations)
    ])
    bb_xyz[:, 0] = np.clip(bb_xyz[:, 0], 0.5, 28.5)
    bb_xyz[:, 1] = np.clip(bb_xyz[:, 1], 0.5, 28.5)

    solver = EikonalSolver(vm)
    tt_das = solver.travel_times_at_receivers(true_source, das_xyz, 'P')
    tt_bb = solver.travel_times_at_receivers(true_source, bb_xyz, 'P')
    true_tau = 5.0

    das_sigma = 0.003 * noise_level
    das_corr = 0.05
    bb_sigmas = 0.02 + 0.06 * rng.rand(n_bb_stations)

    das_picks = DASPicks(
        times=tt_das + true_tau + das_sigma * rng.randn(n_das_channels),
        receiver_xyz=das_xyz, fiber_coords=fiber_xi, phase='P',
        pick_sigma=das_sigma, clock_sigma=0.001,
        correlation_length=das_corr,
    )
    bb_picks = BroadbandPicks(
        times=tt_bb + true_tau + bb_sigmas * rng.randn(n_bb_stations) * noise_level,
        receiver_xyz=bb_xyz, phases=['P'] * n_bb_stations,
        uncertainties=bb_sigmas,
    )

    return {
        'das_picks': das_picks,
        'bb_picks': bb_picks,
        'velocity_model': vm,
        'true_source': true_source,
        'true_origin_time': true_tau,
        'das_xyz': das_xyz,
        'bb_xyz': bb_xyz,
    }


def generate_alaska_scenario(seed: int = 42) -> Dict[str, Any]:
    """Alaska TAPS corridor: 27 km fiber + regional broadband network."""
    rng = np.random.RandomState(seed)

    nx, ny, nz = 60, 60, 20
    vp = np.zeros((nx, ny, nz))
    for iz in range(nz):
        vp[:, :, iz] = 5.0 + 0.08 * iz
    vp[25:35, 28:32, 5:15] *= 0.93
    vs = vp / 1.73

    vm = VelocityModel(vp=vp, vs=vs, origin=np.array([0., 0., 0.]),
                       spacing=np.array([1., 1., 1.]))

    n_das = 500
    fiber_xi = np.linspace(0, 27.0, n_das)
    das_xyz = np.column_stack([
        10.0 + fiber_xi * np.cos(np.radians(15)),
        15.0 + fiber_xi * np.sin(np.radians(15)),
        np.zeros(n_das) + 0.1
    ])

    n_bb = 20
    angles = np.linspace(0, 2 * np.pi, n_bb, endpoint=False)
    radii = np.concatenate([5 + 15 * rng.rand(10), 20 + 30 * rng.rand(10)])
    bb_xyz = np.column_stack([
        30 + radii * np.cos(angles), 30 + radii * np.sin(angles),
        np.zeros(n_bb)
    ])
    bb_xyz[:, 0] = np.clip(bb_xyz[:, 0], 1, 58)
    bb_xyz[:, 1] = np.clip(bb_xyz[:, 1], 1, 58)

    true_source = np.array([23.0, 22.0, 10.0])
    true_tau = 5.0

    solver = EikonalSolver(vm)
    tt_das = solver.travel_times_at_receivers(true_source, das_xyz, 'P')
    tt_bb = solver.travel_times_at_receivers(true_source, bb_xyz, 'P')

    das_sigma, das_corr, clock_sigma = 0.003, 0.05, 0.001
    bb_sigmas = 0.02 + 0.08 * rng.rand(n_bb)

    das_picks = DASPicks(
        times=tt_das + true_tau + das_sigma * rng.randn(n_das),
        receiver_xyz=das_xyz, fiber_coords=fiber_xi, phase='P',
        pick_sigma=das_sigma, clock_sigma=clock_sigma,
        correlation_length=das_corr,
    )
    bb_picks = BroadbandPicks(
        times=tt_bb + true_tau + bb_sigmas * rng.randn(n_bb),
        receiver_xyz=bb_xyz, phases=['P'] * n_bb, uncertainties=bb_sigmas,
    )

    return {
        'das_picks': das_picks, 'bb_picks': bb_picks,
        'velocity_model': vm, 'true_source': true_source,
        'true_origin_time': true_tau, 'das_xyz': das_xyz, 'bb_xyz': bb_xyz,
    }


def generate_ocean_island_scenario(seed: int = 99) -> Dict[str, Any]:
    """Volcanic DAS: 10 km fiber on flank + monitoring network."""
    rng = np.random.RandomState(seed)

    nx, ny, nz = 40, 40, 15
    dx = dy = dz = 0.5
    vp = np.zeros((nx, ny, nz))
    for iz in range(nz):
        vp[:, :, iz] = 3.5 + 0.3 * iz
    cx, cy, cz = 20, 20, 6
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                r = np.sqrt((ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2) * dx
                if r < 2.0:
                    vp[ix, iy, iz] *= (0.75 + 0.25 * (r / 2.0))
    vs = vp / 1.78

    vm = VelocityModel(vp=vp, vs=vs, origin=np.array([0., 0., 0.]),
                       spacing=np.array([dx, dy, dz]))

    n_das = 500
    fiber_xi = np.linspace(0, 10.0, n_das)
    das_xyz = np.column_stack([
        5.0 + fiber_xi * 0.9,
        8.0 + fiber_xi * 0.1 + 0.3 * np.sin(fiber_xi * 0.5),
        np.zeros(n_das) + 0.05
    ])
    das_xyz[:, 0] = np.clip(das_xyz[:, 0], 0.5, 19.0)
    das_xyz[:, 1] = np.clip(das_xyz[:, 1], 0.5, 19.0)

    n_bb = 25
    angles = np.linspace(0, 2*np.pi, n_bb, endpoint=False) + rng.uniform(-0.2, 0.2, n_bb)
    radii = 2.0 + 6.0 * rng.rand(n_bb)
    bb_xyz = np.column_stack([
        10.0 + radii * np.cos(angles), 10.0 + radii * np.sin(angles),
        np.zeros(n_bb)
    ])
    bb_xyz[:, 0] = np.clip(bb_xyz[:, 0], 0.5, 19.0)
    bb_xyz[:, 1] = np.clip(bb_xyz[:, 1], 0.5, 19.0)

    true_source = np.array([10.0, 10.0, 3.0])
    true_tau = 2.0

    solver = EikonalSolver(vm)
    tt_das = solver.travel_times_at_receivers(true_source, das_xyz, 'P')
    tt_bb = solver.travel_times_at_receivers(true_source, bb_xyz, 'P')

    das_sigma, das_corr = 0.005, 0.03
    bb_sigmas = 0.01 + 0.04 * rng.rand(n_bb)

    das_picks = DASPicks(
        times=tt_das + true_tau + das_sigma * rng.randn(n_das),
        receiver_xyz=das_xyz, fiber_coords=fiber_xi, phase='P',
        pick_sigma=das_sigma, clock_sigma=0.002,
        correlation_length=das_corr,
    )
    bb_picks = BroadbandPicks(
        times=tt_bb + true_tau + bb_sigmas * rng.randn(n_bb),
        receiver_xyz=bb_xyz, phases=['P'] * n_bb, uncertainties=bb_sigmas,
    )

    return {
        'das_picks': das_picks, 'bb_picks': bb_picks,
        'velocity_model': vm, 'true_source': true_source,
        'true_origin_time': true_tau, 'das_xyz': das_xyz, 'bb_xyz': bb_xyz,
    }
