"""
Optimal sensor placement using Fisher information.

D/A/E-optimal criteria, greedy sensor addition,
geometry weight mapping, and azimuthal gap analysis.
"""

import numpy as np
from scipy.linalg import inv
from scipy.optimize import nnls
from typing import Optional, Dict, Any, List
import logging

from .data import VelocityModel
from .forward import EikonalSolver

logger = logging.getLogger(__name__)

class OptimalDesign:
    """
    Optimal experimental design for seismic networks.

    Given a velocity model and a set of target source locations,
    compute where to place sensors to maximize information about
    either earthquake location or velocity structure.

    The core idea: the Fisher information matrix F depends on
    receiver geometry through the Fréchet derivatives G. We can
    evaluate F for any candidate receiver configuration without
    needing actual data — only the forward model geometry matters.
    """

    def __init__(self, velocity_model: VelocityModel):
        self.vm = velocity_model
        self.solver = EikonalSolver(velocity_model)

    def compute_station_fisher(
        self,
        source_xyz: np.ndarray,
        receiver_xyz: np.ndarray,
        phase: str,
        sigma: float,
    ) -> np.ndarray:
        """
        Fisher information contributed by a single station (or group)
        for the location problem.

        F_station = G^T (σ²I)^{-1} G = G^T G / σ²

        Parameters
        ----------
        source_xyz : (3,) source location
        receiver_xyz : (n, 3) receiver positions
        phase : 'P' or 'S'
        sigma : pick uncertainty (s)

        Returns
        -------
        F : (4, 4) Fisher information matrix for (x, y, z, τ)
        """
        G_src = self.solver.frechet_source(source_xyz, receiver_xyz, phase)
        n = len(receiver_xyz)
        G = np.hstack([G_src, np.ones((n, 1))])
        return G.T @ G / sigma ** 2

    def d_optimal_criterion(self, F: np.ndarray) -> float:
        """
        D-optimal criterion: log det(F).
        Maximizing this minimizes the volume of the uncertainty ellipsoid.
        Equivalent to maximizing the product of all eigenvalues of F.
        """
        eigvals = np.linalg.eigvalsh(F)
        # Use log for numerical stability
        positive = eigvals[eigvals > 1e-15]
        if len(positive) < F.shape[0]:
            return -np.inf  # singular — infinite uncertainty
        return np.sum(np.log(positive))

    def a_optimal_criterion(self, F: np.ndarray) -> float:
        """
        A-optimal criterion: -tr(F⁻¹).
        Maximizing this minimizes the average variance across all parameters.
        Negative because we want to maximize (minimize tr(F⁻¹)).
        """
        try:
            C = inv(F)
            return -np.trace(C)
        except np.linalg.LinAlgError:
            return -np.inf

    def e_optimal_criterion(self, F: np.ndarray) -> float:
        """
        E-optimal criterion: min eigenvalue of F.
        Maximizing this minimizes the worst-case uncertainty direction.
        """
        return float(np.min(np.linalg.eigvalsh(F)))

    # --------------------------------------------------------
    # GREEDY SENSOR ADDITION
    # --------------------------------------------------------

    def greedy_optimal_placement(
        self,
        target_sources: np.ndarray,
        candidate_positions: np.ndarray,
        existing_F: np.ndarray,
        n_to_add: int,
        phase: str = 'P',
        sigma: float = 0.05,
        criterion: str = 'D',
        task: str = 'location',
    ) -> Dict[str, Any]:
        """
        Greedily add sensors to maximize information.

        At each step, evaluate every candidate position and pick
        the one that most improves the design criterion. This is
        a (1-1/e)-approximate solution to the NP-hard optimal
        subset selection problem (submodularity of log det).

        Parameters
        ----------
        target_sources : (n_sources, 3) source locations to optimize for
        candidate_positions : (n_candidates, 3) potential sensor locations
        existing_F : (4,4) or (n_model, n_model) current Fisher from
                     existing network
        n_to_add : how many sensors to add
        phase : seismic phase
        sigma : assumed pick uncertainty for new sensors
        criterion : 'D' (volume), 'A' (average), or 'E' (worst-case)
        task : 'location' or 'tomography'

        Returns
        -------
        dict with:
          selected_indices : indices into candidate_positions
          selected_positions : (n_to_add, 3)
          criterion_history : criterion value after each addition
          marginal_gains : information gain from each addition
          F_final : final Fisher information matrix
        """
        crit_fn = {
            'D': self.d_optimal_criterion,
            'A': self.a_optimal_criterion,
            'E': self.e_optimal_criterion,
        }[criterion]

        F_current = existing_F.copy()
        n_params = F_current.shape[0]

        selected = []
        criterion_history = [crit_fn(F_current)]
        marginal_gains = []
        available = set(range(len(candidate_positions)))

        for step in range(n_to_add):
            best_idx = None
            best_crit = -np.inf
            best_dF = None

            for idx in available:
                pos = candidate_positions[idx:idx + 1]

                if task == 'location':
                    # Average Fisher over all target sources
                    dF = np.zeros((n_params, n_params))
                    for src in target_sources:
                        dF += self.compute_station_fisher(
                            src, pos, phase, sigma
                        )
                    dF /= len(target_sources)
                else:
                    # Tomography: use velocity Fréchet kernel
                    dF = np.zeros((n_params, n_params))
                    for src in target_sources:
                        G_m = self.solver.frechet_model(src, pos, phase)
                        dF += (G_m.T @ G_m / sigma ** 2).toarray()
                    dF /= len(target_sources)

                F_trial = F_current + dF
                crit_val = crit_fn(F_trial)

                if crit_val > best_crit:
                    best_crit = crit_val
                    best_idx = idx
                    best_dF = dF

            if best_idx is None:
                logger.warning(f"No improving candidate at step {step}")
                break

            selected.append(best_idx)
            available.remove(best_idx)
            gain = best_crit - criterion_history[-1]
            marginal_gains.append(gain)
            F_current = F_current + best_dF
            criterion_history.append(best_crit)

            logger.info(
                f"  Step {step + 1}: added candidate {best_idx}, "
                f"criterion = {best_crit:.4f}, gain = {gain:.4f}"
            )

        return {
            'selected_indices': selected,
            'selected_positions': candidate_positions[selected],
            'criterion_history': criterion_history,
            'marginal_gains': marginal_gains,
            'F_final': F_current,
        }

    # --------------------------------------------------------
    # WEIGHT MAPPING: NON-IDEAL → OPTIMAL PROXY
    # --------------------------------------------------------

    def compute_geometry_weights(
        self,
        source_xyz: np.ndarray,
        actual_receivers: np.ndarray,
        phase: str,
        sigmas: np.ndarray,
        target_F: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute per-station weights that map a non-ideal network
        geometry to approximate an optimal (isotropic) Fisher matrix.

        The idea: given actual receivers with Fisher F_actual, find
        weights w_i such that Σ w_i F_i ≈ F_target, where F_target
        is either provided or set to a scaled identity (isotropic).

        This is solved as a non-negative least squares problem
        on the vectorized Fisher matrices.

        Parameters
        ----------
        source_xyz : (3,) source location
        actual_receivers : (n, 3) actual station positions
        phase : 'P' or 'S'
        sigmas : (n,) per-station uncertainties
        target_F : (4,4) desired Fisher matrix. If None, uses
                   scaled identity (isotropic resolution).

        Returns
        -------
        weights : (n,) per-station weights
        """
        n = len(actual_receivers)

        # Compute per-station Fisher matrices
        station_Fs = []
        for i in range(n):
            F_i = self.compute_station_fisher(
                source_xyz, actual_receivers[i:i + 1], phase, sigmas[i]
            )
            station_Fs.append(F_i)

        if target_F is None:
            # Isotropic target: scaled identity with same total information
            F_total = sum(station_Fs)
            target_scale = np.trace(F_total) / 4.0
            target_F = target_scale * np.eye(4)

        # Vectorize: each F_i → 10-element vector (upper triangle)
        triu_idx = np.triu_indices(4)

        A = np.zeros((len(triu_idx[0]), n))
        for i, F_i in enumerate(station_Fs):
            A[:, i] = F_i[triu_idx]

        b = target_F[triu_idx]

        # Non-negative least squares
        from scipy.optimize import nnls
        weights, residual = nnls(A, b)

        # Normalize so mean weight = 1
        if np.sum(weights) > 0:
            weights *= n / np.sum(weights)

        return weights

    def azimuthal_gap_analysis(
        self,
        source_xyz: np.ndarray,
        receiver_xyz: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute azimuthal coverage metrics.

        Large azimuthal gaps correlate with poorly constrained
        horizontal location. This is a fast proxy for when you
        can't afford full Fisher analysis.

        Returns
        -------
        dict with:
          max_gap_deg : largest azimuthal gap
          secondary_gap_deg : second largest gap
          n_quadrants : number of occupied azimuthal quadrants (0-4)
          azimuthal_uniformity : 0 (all in one direction) to 1 (uniform)
        """
        dx = receiver_xyz[:, 0] - source_xyz[0]
        dy = receiver_xyz[:, 1] - source_xyz[1]
        azimuths = np.degrees(np.arctan2(dx, dy)) % 360
        azimuths = np.sort(azimuths)

        if len(azimuths) < 2:
            return {
                'max_gap_deg': 360.0,
                'secondary_gap_deg': 360.0,
                'n_quadrants': int(len(azimuths)),
                'azimuthal_uniformity': 0.0,
            }

        gaps = np.diff(azimuths)
        gaps = np.append(gaps, 360.0 - azimuths[-1] + azimuths[0])
        sorted_gaps = np.sort(gaps)[::-1]

        quadrants = set((azimuths // 90).astype(int))

        # Uniformity: compare to ideal uniform distribution
        n = len(azimuths)
        ideal_gap = 360.0 / n
        uniformity = 1.0 - np.std(gaps) / ideal_gap if n > 1 else 0.0
        uniformity = max(0.0, min(1.0, uniformity))

        return {
            'max_gap_deg': float(sorted_gaps[0]),
            'secondary_gap_deg': float(sorted_gaps[1]) if len(sorted_gaps) > 1 else 0.0,
            'n_quadrants': len(quadrants),
            'azimuthal_uniformity': uniformity,
        }

