"""
Data structures for DAS and broadband seismic picks.

Each class knows how to build its own covariance matrix,
which encodes the statistical structure of the measurement errors.
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DASPicks:
    """
    Collection of DAS arrival-time picks with fiber metadata.

    The covariance has three components:
      1. Along-fiber spatial correlation (exponential kernel)
      2. Shared interrogator clock jitter (rank-1)
      3. Independent per-channel pick noise (diagonal)

    Parameters
    ----------
    times : (N,) observed arrival times (s)
    receiver_xyz : (N, 3) channel positions (km)
    fiber_coords : (N,) along-fiber distance ξ (km)
    phase : 'P' or 'S'
    pick_sigma : per-channel uncertainty (s), default 2 ms
    clock_sigma : shared clock jitter (s), default 1 ms
    correlation_length : along-fiber correlation (km), default 0.05 = 50 m
    channel_spacing : physical spacing (km), default 0.001 = 1 m
    """
    times: np.ndarray
    receiver_xyz: np.ndarray
    fiber_coords: np.ndarray
    phase: str
    pick_sigma: float = 0.002
    clock_sigma: float = 0.001
    correlation_length: float = 0.05
    channel_spacing: float = 0.001

    @property
    def n_picks(self) -> int:
        return len(self.times)

    def build_covariance(self) -> np.ndarray:
        """
        Full NxN covariance matrix.

        C_ij = σ²_das exp(-|ξ_i - ξ_j|² / 2ℓ²)
             + σ²_clock
             + δ_ij σ²_pick

        WARNING: O(N²) memory. For N > 10000, use build_covariance_lowrank.
        """
        xi = self.fiber_coords
        dist_sq = (xi[:, None] - xi[None, :]) ** 2
        C = (self.pick_sigma ** 2) * np.exp(
            -dist_sq / (2.0 * self.correlation_length ** 2)
        )
        C += self.clock_sigma ** 2
        C += np.diag(np.full(self.n_picks, self.pick_sigma ** 2))
        C += 1e-10 * np.eye(self.n_picks)
        return C

    def build_covariance_lowrank(
        self, n_components: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Low-rank approximation: C ≈ U Λ U^T + σ²_pick I.
        Returns (eigenvalues, eigenvectors) of correlated part.
        """
        C = self.build_covariance()
        C_corr = C - np.diag(np.full(self.n_picks, self.pick_sigma ** 2))
        eigvals, eigvecs = eigh(
            C_corr,
            subset_by_index=[self.n_picks - n_components, self.n_picks - 1]
        )
        return eigvals, eigvecs

    def effective_n(self, C: Optional[np.ndarray] = None) -> float:
        """
        Effective independent observations: N_eff = tr(C)² / tr(C²).
        Typically N_eff ≈ 200-500 for 5000 channels at 1m/50m spacing.
        """
        if C is None:
            C = self.build_covariance()
        tr = np.trace(C)
        tr2 = np.trace(C @ C)
        return tr ** 2 / tr2


@dataclass
class BroadbandPicks:
    """
    Broadband network picks — approximately independent,
    heteroscedastic (quality-dependent uncertainty).

    Parameters
    ----------
    times : (N,) arrival times (s)
    receiver_xyz : (N, 3) station positions (km)
    phases : list of 'P' or 'S'
    uncertainties : (N,) per-pick 1-sigma (s), typically 0.01–0.2
    """
    times: np.ndarray
    receiver_xyz: np.ndarray
    phases: List[str]
    uncertainties: np.ndarray

    @property
    def n_picks(self) -> int:
        return len(self.times)

    def build_covariance(self) -> np.ndarray:
        return np.diag(self.uncertainties ** 2)

    def build_covariance_inverse(self) -> np.ndarray:
        return np.diag(1.0 / self.uncertainties ** 2)


@dataclass
class VelocityModel:
    """
    3D velocity model on a regular Cartesian grid.

    Parameters
    ----------
    vp : (nx, ny, nz) P-wave velocity (km/s)
    vs : (nx, ny, nz) S-wave velocity (km/s)
    origin : (3,) grid origin [x0, y0, z0] (km)
    spacing : (3,) grid spacing [dx, dy, dz] (km)
    """
    vp: np.ndarray
    vs: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.vp.shape

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def slowness(self, phase: str) -> np.ndarray:
        if phase == 'P':
            return 1.0 / self.vp
        elif phase == 'S':
            return 1.0 / self.vs
        raise ValueError(f"Unknown phase: {phase}")

    def grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        x = self.origin[0] + np.arange(nx) * self.spacing[0]
        y = self.origin[1] + np.arange(ny) * self.spacing[1]
        z = self.origin[2] + np.arange(nz) * self.spacing[2]
        return x, y, z
