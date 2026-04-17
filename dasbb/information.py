"""
Task-specific information gain metrics.

Location gain via KL divergence, per-station marginal value,
and tomography resolution gain. Key insight: these give
DIFFERENT station rankings — a station great for location
may be mediocre for tomography.
"""

import numpy as np
from scipy.linalg import inv
from scipy import sparse
from scipy.sparse.linalg import lsqr
from typing import Optional, List, Tuple, Dict, Any
import logging

from .data import VelocityModel
from .forward import EikonalSolver
from .design import OptimalDesign

logger = logging.getLogger(__name__)

class InformationGain:
    """
    Task-specific information gain metrics.

    For LOCATION: information gain is measured as the KL divergence
    from prior to posterior on the 4D hypocenter parameters.

    For TOMOGRAPHY: information gain is measured as the resolution
    improvement per model cell — how much does adding a data type
    improve the diagonal of the resolution matrix?

    Key insight: these criteria give DIFFERENT sensor rankings.
    A station that's great for location (diverse azimuth) may be
    mediocre for tomography (doesn't illuminate new ray paths).
    """

    def __init__(self, velocity_model: VelocityModel):
        self.vm = velocity_model
        self.solver = EikonalSolver(velocity_model)

    # --------------------------------------------------------
    # LOCATION INFORMATION GAIN
    # --------------------------------------------------------

    def location_kl_divergence(
        self,
        F_prior: np.ndarray,
        F_posterior: np.ndarray,
    ) -> float:
        """
        KL divergence from Gaussian prior to Gaussian posterior
        for the location problem.

        D_KL(posterior || prior) = ½[tr(Σ_prior⁻¹ Σ_post) - k
                                     + ln(det Σ_prior / det Σ_post)]

        where Σ = F⁻¹. For our problem k=4 (x, y, z, τ).

        A larger D_KL means the data told us more — bigger
        reduction from prior uncertainty to posterior uncertainty.
        """
        k = F_prior.shape[0]
        try:
            C_prior = inv(F_prior)
            C_post = inv(F_posterior)
        except np.linalg.LinAlgError:
            return 0.0

        # More numerically stable via eigenvalues
        eigvals_prior = np.linalg.eigvalsh(C_prior)
        eigvals_post = np.linalg.eigvalsh(C_post)

        if np.any(eigvals_prior <= 0) or np.any(eigvals_post <= 0):
            return 0.0

        # tr(Σ_prior⁻¹ Σ_post)
        trace_term = np.trace(F_prior @ C_post)

        # ln(det Σ_prior / det Σ_post) = ln det Σ_prior - ln det Σ_post
        logdet_ratio = (
            np.sum(np.log(eigvals_prior)) -
            np.sum(np.log(eigvals_post))
        )

        kl = 0.5 * (trace_term - k + logdet_ratio)
        return max(0.0, float(kl))

    def location_information_gain_per_station(
        self,
        source_xyz: np.ndarray,
        receiver_xyz: np.ndarray,
        phases: List[str],
        sigmas: np.ndarray,
        F_base: np.ndarray,
    ) -> np.ndarray:
        """
        Marginal location information gain from each station.

        For each station i, compute the KL divergence between
        the posterior with all stations and the posterior with
        station i removed (leave-one-out).

        Stations with high marginal gain are critical for location;
        stations with low gain are redundant.

        Parameters
        ----------
        source_xyz : (3,) source
        receiver_xyz : (n, 3) all station positions
        phases : list of phase types
        sigmas : (n,) per-station uncertainty
        F_base : (4,4) prior Fisher (e.g. from DAS alone)

        Returns
        -------
        gains : (n,) KL divergence gain per station (nats)
        """
        n = len(receiver_xyz)
        od = OptimalDesign(self.vm)

        # Compute per-station Fisher
        station_Fs = []
        for i in range(n):
            F_i = od.compute_station_fisher(
                source_xyz, receiver_xyz[i:i + 1], phases[i], sigmas[i]
            )
            station_Fs.append(F_i)

        F_total = F_base + sum(station_Fs)

        gains = np.zeros(n)
        for i in range(n):
            F_without_i = F_total - station_Fs[i]
            gains[i] = self.location_kl_divergence(F_without_i, F_total)

        return gains

    # --------------------------------------------------------
    # TOMOGRAPHY INFORMATION GAIN
    # --------------------------------------------------------

    def tomography_resolution_gain(
        self,
        source_xyz: np.ndarray,
        receiver_xyz_A: np.ndarray,
        phase_A: str,
        sigma_A: float,
        receiver_xyz_B: np.ndarray,
        phase_B: str,
        sigma_B: float,
        damping: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Resolution improvement from adding data type B to data type A.

        Computes the diagonal of the resolution matrix R for:
          - A alone
          - B alone
          - A + B combined

        The gain is: R_combined - R_A (per model cell).

        This tells you WHERE in the velocity model the additional
        data type helps.

        Returns
        -------
        dict with:
          R_A : (nx, ny, nz) resolution from A alone
          R_B : (nx, ny, nz) resolution from B alone
          R_combined : (nx, ny, nz) resolution from A+B
          gain_from_B : (nx, ny, nz) R_combined - R_A
          gain_from_A : (nx, ny, nz) R_combined - R_B
        """
        n_model = self.vm.n_cells

        # Compute sensitivity matrices
        G_A = self.solver.frechet_model(source_xyz, receiver_xyz_A, phase_A)
        G_B = self.solver.frechet_model(source_xyz, receiver_xyz_B, phase_B)

        # G^T G for each
        GtG_A = (G_A.T @ G_A / sigma_A ** 2)
        GtG_B = (G_B.T @ G_B / sigma_B ** 2)
        GtG_combined = GtG_A + GtG_B

        reg = damping * sparse.eye(n_model)

        def resolution_diagonal(GtG):
            """Stochastic estimation of diag(GtG (GtG + λI)⁻¹)."""
            A = GtG + reg
            n_probes = 20
            R_diag = np.zeros(n_model)
            for _ in range(n_probes):
                z = np.random.choice([-1.0, 1.0], size=n_model)
                Az = lsqr(A.tocsr(), GtG @ z, iter_lim=100)[0]
                R_diag += z * Az
            R_diag /= n_probes
            return np.clip(R_diag, 0, 1).reshape(self.vm.shape)

        R_A = resolution_diagonal(GtG_A)
        R_B = resolution_diagonal(GtG_B)
        R_combined = resolution_diagonal(GtG_combined)

        return {
            'R_A': R_A,
            'R_B': R_B,
            'R_combined': R_combined,
            'gain_from_B': R_combined - R_A,
            'gain_from_A': R_combined - R_B,
        }

    def per_station_tomography_value(
        self,
        sources: np.ndarray,
        receiver_xyz: np.ndarray,
        phases: List[str],
        sigmas: np.ndarray,
        damping: float = 1.0,
        target_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Rank stations by their contribution to tomographic resolution.

        For each station, compute the sum of resolution diagonal
        entries in the target region with vs without that station.

        Parameters
        ----------
        sources : (n_sources, 3) event locations
        receiver_xyz : (n_stations, 3)
        phases, sigmas : per-station
        damping : regularization
        target_region : optional (min_xyz, max_xyz) bounding box
                        to focus on. If None, uses full model.

        Returns
        -------
        values : (n_stations,) total resolution contribution
        """
        n_model = self.vm.n_cells
        n_stations = len(receiver_xyz)

        # Build total G^T G from all stations and sources
        GtG_total = sparse.lil_matrix((n_model, n_model))
        per_station_GtG = [
            sparse.lil_matrix((n_model, n_model))
            for _ in range(n_stations)
        ]

        for src in sources:
            for i in range(n_stations):
                G_i = self.solver.frechet_model(
                    src, receiver_xyz[i:i + 1], phases[i]
                )
                contrib = G_i.T @ G_i / sigmas[i] ** 2
                GtG_total += contrib
                per_station_GtG[i] += contrib

        GtG_total = GtG_total.tocsr()
        reg = damping * sparse.eye(n_model)

        # Mask for target region
        if target_region is not None:
            min_xyz, max_xyz = target_region
            x, y, z = self.vm.grid_coordinates()
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            mask = (
                (xx >= min_xyz[0]) & (xx <= max_xyz[0]) &
                (yy >= min_xyz[1]) & (yy <= max_xyz[1]) &
                (zz >= min_xyz[2]) & (zz <= max_xyz[2])
            ).ravel()
        else:
            mask = np.ones(n_model, dtype=bool)

        # For each station, approximate Δ resolution via rank-1 update
        # This is much cheaper than recomputing the full resolution
        values = np.zeros(n_stations)

        # Total resolution (reference)
        n_probes = 15
        A_total = GtG_total + reg
        R_total = np.zeros(n_model)
        probe_vectors = [
            np.random.choice([-1.0, 1.0], size=n_model)
            for _ in range(n_probes)
        ]
        probe_solutions = []
        for z_vec in probe_vectors:
            sol = lsqr(A_total.tocsr(), GtG_total @ z_vec, iter_lim=100)[0]
            R_total += z_vec * sol
            probe_solutions.append(sol)
        R_total /= n_probes
        R_total_sum = np.sum(np.clip(R_total, 0, 1)[mask])

        for i in range(n_stations):
            GtG_without = GtG_total - per_station_GtG[i].tocsr()
            A_without = GtG_without + reg

            R_without = np.zeros(n_model)
            for z_vec in probe_vectors:
                sol = lsqr(A_without.tocsr(), GtG_without @ z_vec, iter_lim=100)[0]
                R_without += z_vec * sol
            R_without /= n_probes

            R_without_sum = np.sum(np.clip(R_without, 0, 1)[mask])
            values[i] = R_total_sum - R_without_sum

        return values

