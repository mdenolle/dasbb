"""
Joint inversion engine for earthquake location and velocity tomography.

Properly handles correlated DAS noise and heteroscedastic broadband picks
through covariance-weighted Gauss-Newton with Fisher information decomposition.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.stats import chi2 as chi2_dist
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import logging

from .data import DASPicks, BroadbandPicks, VelocityModel
from .forward import EikonalSolver

logger = logging.getLogger(__name__)


@dataclass
class InversionConfig:
    """Inversion configuration parameters."""
    max_iter_location: int = 20
    max_iter_tomography: int = 10
    damping_location: float = 0.1
    damping_model: float = 1.0
    smoothing_weight: float = 0.5
    convergence_tol: float = 1e-4
    compute_posterior: bool = True


class JointInversion:
    """
    Joint DAS + Broadband earthquake location and tomography.

    The covariance-weighted objective function automatically balances
    the two data types without ad hoc reweighting.
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        config: InversionConfig = None
    ):
        self.vm = velocity_model
        self.config = config or InversionConfig()
        self.solver = EikonalSolver(velocity_model)

    def locate_event(
        self,
        das_picks: Optional[DASPicks],
        bb_picks: Optional[BroadbandPicks],
        initial_source: np.ndarray,
        initial_origin_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Locate a single event with full posterior uncertainty.

        Returns dict with source_xyz, origin_time, posterior_cov,
        F_das, F_bb, chi2, location_ellipsoid.
        """
        src = initial_source.copy().astype(np.float64)
        tau = float(initial_origin_time)

        # Precompute covariance factorizations
        C_das_cho = None
        n_eff_das = 0
        if das_picks is not None:
            C_das = das_picks.build_covariance()
            C_das_cho = cho_factor(C_das)
            n_eff_das = das_picks.effective_n(C_das)
            logger.info(
                f"DAS: {das_picks.n_picks} channels, N_eff = {n_eff_das:.0f}, "
                f"correlation_length = {das_picks.correlation_length*1000:.0f} m"
            )

        C_bb_inv = None
        if bb_picks is not None:
            C_bb_inv = bb_picks.build_covariance_inverse()
            logger.info(
                f"Broadband: {bb_picks.n_picks} picks, "
                f"σ range = [{bb_picks.uncertainties.min():.3f}, "
                f"{bb_picks.uncertainties.max():.3f}] s"
            )

        F_das_mat = np.zeros((4, 4))
        F_bb_mat = np.zeros((4, 4))
        r_das = r_bb = None

        for iteration in range(self.config.max_iter_location):
            H = np.zeros((4, 4))
            grad = np.zeros(4)
            F_das_mat[:] = 0
            F_bb_mat[:] = 0

            # DAS contribution
            if das_picks is not None:
                tt_das = self.solver.travel_times_at_receivers(
                    src, das_picks.receiver_xyz, das_picks.phase
                )
                r_das = das_picks.times - (tt_das + tau)
                G_das_src = self.solver.frechet_source(
                    src, das_picks.receiver_xyz, das_picks.phase
                )
                G_das = np.hstack([G_das_src, np.ones((das_picks.n_picks, 1))])
                CinvG = cho_solve(C_das_cho, G_das)
                Cinvr = cho_solve(C_das_cho, r_das)
                F_das_mat = G_das.T @ CinvG
                H += F_das_mat
                grad += G_das.T @ Cinvr

            # Broadband contribution (batched by phase)
            if bb_picks is not None:
                n_bb = bb_picks.n_picks
                phase_groups = {}
                for j in range(n_bb):
                    ph = bb_picks.phases[j]
                    phase_groups.setdefault(ph, []).append(j)

                tt_bb = np.zeros(n_bb)
                G_bb_src = np.zeros((n_bb, 3))
                for ph, indices in phase_groups.items():
                    idx_arr = np.array(indices)
                    rx_batch = bb_picks.receiver_xyz[idx_arr]
                    tt_bb[idx_arr] = self.solver.travel_times_at_receivers(src, rx_batch, ph)
                    G_bb_src[idx_arr] = self.solver.frechet_source(src, rx_batch, ph)

                r_bb = bb_picks.times - (tt_bb + tau)
                G_bb = np.hstack([G_bb_src, np.ones((n_bb, 1))])
                CinvG_bb = C_bb_inv @ G_bb
                Cinvr_bb = C_bb_inv @ r_bb
                F_bb_mat = G_bb.T @ CinvG_bb
                H += F_bb_mat
                grad += G_bb.T @ Cinvr_bb

            # Marquardt-damped Gauss-Newton step with step limiting
            lam = self.config.damping_location
            H_damped = H + lam * np.diag(np.diag(H) + 1e-10)
            delta = np.linalg.solve(H_damped, grad)

            max_step = 2.0 * float(np.max(self.vm.spacing))
            spatial_step = np.linalg.norm(delta[:3])
            if spatial_step > max_step:
                delta *= max_step / spatial_step

            src += delta[:3]
            tau += delta[3]

            # Clamp source to stay within velocity model grid
            grid_min = self.vm.origin + self.vm.spacing
            grid_max = self.vm.origin + (np.array(self.vm.shape) - 2) * self.vm.spacing
            src = np.clip(src, grid_min, grid_max)

            conv_thresh = max(self.config.convergence_tol,
                              0.01 * float(np.min(self.vm.spacing)))
            if np.linalg.norm(delta[:3]) < conv_thresh:
                logger.info(f"Location converged at iteration {iteration}")
                break

        # Posterior analysis
        C_post = None
        ellipsoid = None
        if self.config.compute_posterior:
            try:
                C_post = np.linalg.inv(H)
                ellipsoid = self._compute_ellipsoid(C_post)
            except np.linalg.LinAlgError:
                logger.warning("Singular Hessian — posterior not available")

        chi2 = {}
        if das_picks is not None and r_das is not None:
            chi2_das = float(r_das @ cho_solve(C_das_cho, r_das))
            chi2['das'] = chi2_das
            chi2['das_per_neff'] = chi2_das / max(n_eff_das, 1)
        if bb_picks is not None and r_bb is not None:
            chi2_bb = float(r_bb @ C_bb_inv @ r_bb)
            chi2['bb'] = chi2_bb
            chi2['bb_per_n'] = chi2_bb / max(bb_picks.n_picks, 1)

        return {
            'source_xyz': src,
            'origin_time': tau,
            'posterior_cov': C_post,
            'F_das': F_das_mat,
            'F_bb': F_bb_mat,
            'chi2': chi2,
            'n_iterations': iteration + 1,
            'location_ellipsoid': ellipsoid,
        }

    def _compute_ellipsoid(self, C_post, confidence=0.95):
        C_spatial = C_post[:3, :3]
        eigenvalues, eigenvectors = eigh(C_spatial)
        kappa = np.sqrt(chi2_dist.ppf(confidence, df=3))
        semi_axes = kappa * np.sqrt(np.maximum(eigenvalues, 0))
        return {
            'semi_axes_km': semi_axes,
            'rotation_matrix': eigenvectors,
            'confidence': confidence,
            'volume_km3': (4.0 / 3.0) * np.pi * np.prod(semi_axes),
        }

    def information_decomposition(self, F_das, F_bb):
        """
        Decompose Fisher information by data type per eigendirection.
        Answers: in which parameter direction does each data type help?
        """
        F_total = F_das + F_bb
        eigenvalues, eigenvectors = eigh(F_total)
        param_names = ['X', 'Y', 'Z', 'τ']
        contributions = {}

        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            info_das = float(v @ F_das @ v)
            info_bb = float(v @ F_bb @ v)
            info_total = info_das + info_bb
            dominant = param_names[np.argmax(np.abs(v))]

            if info_total > 1e-15:
                contributions[f'dir_{i}_{dominant}'] = {
                    'eigenvalue': float(eigenvalues[i]),
                    'eigenvector': v.tolist(),
                    'dominant_parameter': dominant,
                    'das_fraction': info_das / info_total,
                    'bb_fraction': info_bb / info_total,
                }
        return contributions

    def joint_tomography(
        self,
        events: List[Tuple[Optional[DASPicks], Optional[BroadbandPicks],
                           np.ndarray, float]],
        n_outer_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Joint hypocenter–velocity inversion via alternating optimization.
        """
        if n_outer_iterations is None:
            n_outer_iterations = self.config.max_iter_tomography

        locations = [None] * len(events)
        convergence = []

        for outer in range(n_outer_iterations):
            logger.info(f"=== Tomography iteration {outer+1}/{n_outer_iterations} ===")

            total_chi2 = 0.0
            for i, (das_p, bb_p, init_src, init_tau) in enumerate(events):
                if locations[i] is not None:
                    init_src = locations[i]['source_xyz']
                    init_tau = locations[i]['origin_time']
                result = self.locate_event(das_p, bb_p, init_src, init_tau)
                locations[i] = result
                total_chi2 += sum(v for k, v in result['chi2'].items() if k in ('das', 'bb'))

            convergence.append(total_chi2)
            dm = self._velocity_update(events, locations)

            dm_3d = dm[:self.vm.n_cells].reshape(self.vm.shape)
            vp_vs_ratio = self.vm.vp / self.vm.vs
            self.vm.vp = np.maximum(self.vm.vp + dm_3d, 0.5)
            self.vm.vs = self.vm.vp / vp_vs_ratio
            self.solver = EikonalSolver(self.vm)

        R_diag = self._compute_resolution(events, locations)
        return {
            'locations': locations,
            'velocity_model': self.vm,
            'resolution_diagonal': R_diag,
            'convergence_history': convergence,
        }

    def _velocity_update(self, events, locations):
        n_model = self.vm.n_cells
        G_blocks, rhs_blocks = [], []

        for i, (das_p, bb_p, _, _) in enumerate(events):
            src = locations[i]['source_xyz']
            tau = locations[i]['origin_time']

            if das_p is not None:
                tt = self.solver.travel_times_at_receivers(src, das_p.receiver_xyz, das_p.phase)
                r = das_p.times - (tt + tau)
                G_m = self.solver.frechet_model(src, das_p.receiver_xyz, das_p.phase)
                C_das = das_p.build_covariance()
                eigvals, eigvecs = eigh(C_das)
                Cinv_half = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-15))) @ eigvecs.T
                G_blocks.append(sparse.csr_matrix(Cinv_half @ G_m.toarray()))
                rhs_blocks.append(Cinv_half @ r)

            if bb_p is not None:
                for j in range(bb_p.n_picks):
                    tt_j = self.solver.travel_times_at_receivers(src, bb_p.receiver_xyz[j:j+1], bb_p.phases[j])[0]
                    r_j = bb_p.times[j] - (tt_j + tau)
                    G_m_j = self.solver.frechet_model(src, bb_p.receiver_xyz[j:j+1], bb_p.phases[j])
                    w = 1.0 / bb_p.uncertainties[j]
                    G_blocks.append(w * G_m_j)
                    rhs_blocks.append(np.array([w * r_j]))

        G_full = sparse.vstack(G_blocks)
        rhs = np.concatenate(rhs_blocks)
        L = self._build_3d_laplacian(self.vm.shape)
        G_reg = sparse.vstack([G_full, self.config.smoothing_weight * L])
        rhs_reg = np.concatenate([rhs, np.zeros(L.shape[0])])
        dm = lsqr(G_reg, rhs_reg, damp=self.config.damping_model, iter_lim=500)[0]
        return dm

    @staticmethod
    def _build_3d_laplacian(shape):
        nx, ny, nz = shape
        n = nx * ny * nz
        diags = [-6.0 * np.ones(n)]
        offsets = [0]
        for offset in [1, nz, ny * nz]:
            diags.extend([np.ones(n - offset), np.ones(n - offset)])
            offsets.extend([offset, -offset])
        return sparse.diags(diags, offsets, shape=(n, n), format='csr')

    def _compute_resolution(self, events, locations, n_probes=30):
        n_model = self.vm.n_cells
        GtG = sparse.lil_matrix((n_model, n_model))
        for i, (das_p, bb_p, _, _) in enumerate(events):
            src = locations[i]['source_xyz']
            if das_p is not None:
                G = self.solver.frechet_model(src, das_p.receiver_xyz, das_p.phase)
                GtG += G.T @ G
            if bb_p is not None:
                G = self.solver.frechet_model(src, bb_p.receiver_xyz, bb_p.phases[0])
                GtG += G.T @ G

        GtG = GtG.tocsr()
        reg = self.config.damping_model * sparse.eye(n_model)
        A = GtG + reg

        R_diag = np.zeros(n_model)
        for _ in range(n_probes):
            z = np.random.choice([-1.0, 1.0], size=n_model)
            R_diag += z * lsqr(A, GtG @ z, iter_lim=200)[0]
        return np.clip(R_diag / n_probes, 0, 1).reshape(self.vm.shape)
